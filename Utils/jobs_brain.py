import os
import argparse
import traceback
from pathlib import Path
import pickle
import json
import numpy as np
import scanpy as sc
import anndata as ad

import sys
sys.path.append('../Utils')
from Solve_Slice_test import solve_V_all_slices, find_B_music, select_informative_genes, from_anndata


# -----------------------------
# Data prep (yours, minimal edits)
# -----------------------------
def prepare_data(threshold=1.25, return_sc=False, fname="../../Data/MOSTA/Dorsal_midbrain_cell_bin.h5ad"):
    # fname = "../../Data/MOSTA/Dorsal_midbrain_cell_bin.h5ad"
    adata = sc.read_h5ad(fname)
    adata.X = adata.layers["counts"].copy()

    
    sc.pp.filter_genes(adata, min_counts=1)

    batches_out = ['SS200000131BL_C3C4', 'SS200000131BL_C5C6']
    batches = ['FP200000600TR_E3', 'SS200000108BR_B1B2', 'SS200000108BR_A3A4']

    adata_all = []
    for t in batches:
        print(t)
        idx = adata.obs["Batch"] == t
        print(adata[idx].obs['Time point'][0])
        adata_temp = adata[idx].copy()
        adata_temp.obs['x'] = adata_temp.obsm['spatial'][:, 0]
        adata_temp.obs['y'] = adata_temp.obsm['spatial'][:, 1]
        adata_all.append(adata_temp)

    adata_all_out = []
    for t in batches_out:
        print(t)
        idx = adata.obs["Batch"] == t
        print(adata[idx].obs['Time point'][0])
        adata_temp = adata[idx].copy()
        sc.pp.normalize_total(adata_temp, target_sum=1e4)
        adata_temp.obs['x'] = adata_temp.obsm['spatial'][:, 0]
        adata_temp.obs['y'] = adata_temp.obsm['spatial'][:, 1]
        adata_all_out.append(adata_temp)

    adata_out = ad.concat(adata_all_out)
    adata_out.obs['cellType'] = adata_out.obs['annotation']

    B = find_B_music(adata_out)
    pd_B = B.T

    adata_out.X = adata_out.layers["counts"].copy()
    counts, cell_types = from_anndata(adata_out, celltype_key="cellType")
    genes_subset = select_informative_genes(pd_B, counts, cell_types, fc_thresh_ln=threshold)
    print(len(genes_subset))
    pd_B = pd_B.loc[:, genes_subset]

    # sc.pp.normalize_total(adata_out, target_sum=1e4)
    if return_sc == False:
        return adata_all, pd_B
    else:
        return adata_all, pd_B, adata_out


# -----------------------------
# Save results (yours)
# -----------------------------
def save_ragged_results(out_dir, V, pi, mu, lam, aux=None):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / "V.pkl", "wb") as f:
        pickle.dump(V, f, protocol=pickle.HIGHEST_PROTOCOL)

    with open(out_dir / "pi.pkl", "wb") as f:
        pickle.dump(pi, f, protocol=pickle.HIGHEST_PROTOCOL)

    if aux is not None:
        with open(out_dir / "aux.pkl", "wb") as f:
            pickle.dump(aux, f, protocol=pickle.HIGHEST_PROTOCOL)

    (out_dir / "meta.json").write_text(json.dumps({"mu": float(mu), "lam": float(lam)}, indent=2))
    (out_dir / "_SUCCESS").write_text("ok\n")


# -----------------------------
# Grid helpers
# -----------------------------
def _parse_grid(spec: str, kind: str):
    a, b, n = spec.split(",")
    a, b, n = float(a), float(b), int(n)
    if kind == "log":
        if a <= 0 or b <= 0:
            raise ValueError("log grid requires positive a,b")
        return np.logspace(np.log10(a), np.log10(b), n)
    return np.linspace(a, b, n)


def _out_dir(base_dir: Path, mu: float, lam: float) -> Path:
    return Path(base_dir) / f"mu_{mu:.3e}" / f"lam_{lam:.3e}"


def _load_V_from_dir(d: Path):
    with open(d / "V.pkl", "rb") as f:
        return pickle.load(f)

def _load_pi_from_dir(d: Path):
    with open(d / "pi.pkl", "rb") as f:
        return pickle.load(f)


# -----------------------------
# Baseline first, then grid
# -----------------------------
def ensure_baseline(
    *,
    base_dir: Path,
    adata_all,
    pd_B,
    baseline_mu: float,
    lam: float,
    overwrite_baseline: bool,
    ot_solver_baseline: str,
    alpha: float
):
    d = _out_dir(base_dir, baseline_mu, lam)
    if (d / "_SUCCESS").exists() and not overwrite_baseline:
        print(f"[BASELINE] exists: {d}")
        return

    print(f"[BASELINE] computing mu={baseline_mu:.3e}, lam={lam:.3e}")
    V0, pi0, aux0 = solve_V_all_slices(
        adata_all, pd_B,
        lam=lam, mu=baseline_mu, time=None,
        outer_max=20, niter_max=int(1e3), tol=1e-5,
        eta=1e-3, verbose=False,
        ot_solver=ot_solver_baseline, alpha=alpha
    )
    save_ragged_results(d, V0, pi0, baseline_mu, lam, aux=aux0)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mu_grid", required=True)     # e.g. "1e4,1e6,3"
    p.add_argument("--lam_grid", required=True)    # e.g. "1e0,1e6,7"
    p.add_argument("--mu_grid_kind", choices=["log", "lin"], default="log")
    p.add_argument("--lam_grid_kind", choices=["log", "lin"], default="log")
    p.add_argument("--base_dir", default="results_brain")
    p.add_argument("--baseline_mu", type=float, default=0.0)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--overwrite_baseline", action="store_true")
    p.add_argument("--threshold", type=float, default=1.25)
    p.add_argument("--alpha", type=float, default=0.9)

    # Optional: keep your solver choices explicit
    p.add_argument("--ot_solver_baseline", default="sinkhorn")
    p.add_argument("--ot_solver_main", default="low-rank")

    args = p.parse_args()

    mus = _parse_grid(args.mu_grid, args.mu_grid_kind)
    lams = _parse_grid(args.lam_grid, args.lam_grid_kind)

    base_dir = Path(args.base_dir)
    baseline_mu = float(args.baseline_mu)

    try:
        print("[INFO] prepare_data() once ...")
        adata_all, pd_B = prepare_data(threshold=float(args.threshold))

        # 1) Baselines first (one per lam)
        print(f"[INFO] Precomputing baselines at mu={baseline_mu:.3e} for {len(lams)} lambdas ...")
        for lam in lams:
            ensure_baseline(
                base_dir=base_dir,
                adata_all=adata_all,
                pd_B=pd_B,
                baseline_mu=baseline_mu,
                lam=float(lam),
                overwrite_baseline=args.overwrite_baseline,
                ot_solver_baseline=str(args.ot_solver_main),
                alpha = args.alpha
            )

        # 2) Full grid
        print(f"[INFO] Running full grid: {len(mus)} x {len(lams)} = {len(mus)*len(lams)} runs")
        for mu in mus:
            for lam in lams:
                mu = float(mu)
                lam = float(lam)

                out_dir = _out_dir(base_dir, mu, lam)

                if (out_dir / "_SUCCESS").exists() and not args.overwrite:
                    print(f"[SKIP] {out_dir}")
                    continue

                V_init = None
                pi_init = None
                if mu != baseline_mu:
                    baseline_dir = _out_dir(base_dir, baseline_mu, lam)
                    V_init = _load_V_from_dir(baseline_dir)
                    pi_init = _load_pi_from_dir(baseline_dir)

                print(f"[RUN] mu={mu:.3e}, lam={lam:.3e}")

                if V_init is None:
                    V, pi, aux = solve_V_all_slices(
                        adata_all, pd_B,
                        lam=lam, mu=mu, time=None,
                        outer_max=20, niter_max=int(1e3), tol=1e-5,
                        eta=1e-3, verbose=False,
                        ot_solver=str(args.ot_solver_main),
                        alpha = args.alpha
                    )
                    save_ragged_results(out_dir, V, pi, mu, lam, aux=aux)
                else:
                    V, pi, aux = solve_V_all_slices(
                        adata_all, pd_B,
                        lam=lam, mu=mu, time=None,
                        outer_max=20, niter_max=int(1e3), tol=1e-5,
                        eta=1e-3, verbose=False,
                        ot_solver=str(args.ot_solver_main),
                        V_init=V_init,
                        coupling = pi_init,
                        alpha = args.alpha
                    )

                    save_ragged_results(out_dir, V, None, mu, lam, aux=aux)

    except Exception as e:
        base_dir.mkdir(parents=True, exist_ok=True)
        (base_dir / "_FAILED.txt").write_text(
            f"Error: {repr(e)}\n\nTraceback:\n{traceback.format_exc()}\n"
        )
        raise


if __name__ == "__main__":
    main()

