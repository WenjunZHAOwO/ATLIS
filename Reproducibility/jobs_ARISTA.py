import os
import sys
import argparse
import traceback
from pathlib import Path
import tempfile
import pickle
import json

import numpy as np
import scanpy as sc

sys.path.append('../Utils')
from Solve_Slice_test import find_B_music, solve_V_all_slices, select_informative_genes, from_anndata


def prepare_template(threshold=1.25):
    # Load the dataset.
    dpi2_rep2 = sc.read_h5ad(
        "../../Data/ARISTA/2DPI_2.h5ad"
    )
    dpi2_rep2.obsm["spatial"] = dpi2_rep2.obsm["spatial"] * np.array([-1, -1])
    dpi2_rep2.X = dpi2_rep2.layers["counts"].copy()
    print("loaded DPI2")

    dpi2_rep2 = dpi2_rep2[dpi2_rep2.obs["inj_uninj"] == "inj"]
    dpi2_rep2 = dpi2_rep2[dpi2_rep2.obs["D_V"] == "D"]

    dpi5_rep2 = sc.read_h5ad(
        "../../Data/ARISTA/5DPI_2.h5ad"
    )
    dpi5_rep2.obsm["spatial"] = dpi5_rep2.obsm["spatial"] * np.array([-1, -1])
    dpi5_rep2.X = dpi5_rep2.layers["counts"].copy()
    print("loaded DPI5")

    dpi5_rep2 = dpi5_rep2[dpi5_rep2.obs["inj_uninj"] == "inj"]
    dpi5_rep2 = dpi5_rep2[dpi5_rep2.obs["D_V"] == "D"]

    dpi10_rep2 = sc.read_h5ad(
        "../../Data/ARISTA/10DPI_2.h5ad"
    )
    dpi10_rep2.obsm["spatial"] = dpi10_rep2.obsm["spatial"] * np.array([-1, -1])
    dpi10_rep2.X = dpi10_rep2.layers["counts"].copy()
    dpi10_rep2 = dpi10_rep2[
        np.arange(dpi10_rep2.n_obs) != np.argmin(dpi10_rep2.obsm["spatial"][:, 0])
    ]
    
    print("loaded DPI10")

    dpi10_rep2 = dpi10_rep2[dpi10_rep2.obs["inj_uninj"] == "inj"]
    dpi10_rep2 = dpi10_rep2[dpi10_rep2.obs["D_V"] == "D"]

    dpi15_rep1 = sc.read_h5ad(
        "../../Data/ARISTA/15DPI_1.h5ad"
    )
    dpi15_rep1.obsm["spatial"] = dpi15_rep1.obsm["spatial"] * np.array([-1, -1])
    dpi15_rep1 = dpi15_rep1[
        np.arange(dpi15_rep1.n_obs) != np.argmax(dpi15_rep1.obsm["spatial"][:, 1])
    ]
    dpi15_rep1.X = dpi15_rep1.layers["counts"].copy()
    print("loaded DPI15")

    dpi15_rep1 = dpi15_rep1[dpi15_rep1.obs["inj_uninj"] == "inj"]
    dpi15_rep1 = dpi15_rep1[dpi15_rep1.obs["D_V"] == "D"]

    dpi20_rep1 = sc.read_h5ad(
        "../../Data/ARISTA/20DPI_1.h5ad"
    )
    dpi20_rep1.obsm["spatial"] = dpi20_rep1.obsm["spatial"] * np.array([-1, -1])
    dpi20_rep1.X = dpi20_rep1.layers["counts"].copy()
    dpi20_rep1 = dpi20_rep1[
        np.arange(dpi20_rep1.n_obs) != np.argmin(dpi20_rep1.obsm["spatial"][:, 0])
    ]
   
    print("loaded DPI20")

    dpi20_rep1 = dpi20_rep1[dpi20_rep1.obs["inj_uninj"] == "inj"]
    dpi20_rep1 = dpi20_rep1[dpi20_rep1.obs["D_V"] == "D"]

    

    adata = sc.concat((dpi2_rep2, dpi5_rep2, dpi10_rep2, dpi15_rep1, dpi20_rep1))
    adata.obs_names_make_unique()

    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.filter_genes(adata, min_counts=1)
    sc.pp.calculate_qc_metrics(adata, inplace=True)

    sc.pp.filter_cells(adata, min_genes=200)
    sc.pp.filter_cells(adata, max_genes=adata.obs["n_genes_by_counts"].quantile(0.999))
    sc.pp.filter_genes(adata, min_cells=3)

    adata.obs["x"] = adata.obsm["spatial"][:, 0]
    adata.obs["y"] = adata.obsm["spatial"][:, 1]

    adata.obs = adata.obs.rename(columns={"Annotation": "cellType"})
    
    return adata




def prepare_data(threshold=1.25, return_sc=False):

    # Load the dataset.
    dpi2_rep1 = sc.read_h5ad(
        "../../Data/ARISTA/2DPI_1.h5ad"
    )
    dpi2_rep1.obsm["spatial"] = dpi2_rep1.obsm["spatial"] * np.array([-1, -1])
    dpi2_rep1.X = dpi2_rep1.layers["counts"].copy()
    print("loaded DPI2")

    dpi2_rep1 = dpi2_rep1[dpi2_rep1.obs["inj_uninj"] == "inj"]
    dpi2_rep1 = dpi2_rep1[dpi2_rep1.obs["D_V"] == "D"]

    dpi5_rep1 = sc.read_h5ad(
        "../../Data/ARISTA/5DPI_1.h5ad"
    )
    dpi5_rep1.obsm["spatial"] = dpi5_rep1.obsm["spatial"] * np.array([-1, -1])
    dpi5_rep1.X = dpi5_rep1.layers["counts"].copy()
    print("loaded DPI5")

    dpi5_rep1 = dpi5_rep1[dpi5_rep1.obs["inj_uninj"] == "inj"]
    dpi5_rep1 = dpi5_rep1[dpi5_rep1.obs["D_V"] == "D"]

    dpi10_rep1 = sc.read_h5ad(
        "../../Data/ARISTA/10DPI_1.h5ad"
    )
    dpi10_rep1.obsm["spatial"] = dpi10_rep1.obsm["spatial"] * np.array([-1, -1])
    dpi10_rep1.X = dpi10_rep1.layers["counts"].copy()
    dpi10_rep1 = dpi10_rep1[
        np.arange(dpi10_rep1.n_obs) != np.argmin(dpi10_rep1.obsm["spatial"][:, 0])
    ]
    dpi10_rep1 = dpi10_rep1[
        np.arange(dpi10_rep1.n_obs) != np.argmin(dpi10_rep1.obsm["spatial"][:, 0])
    ]
    dpi10_rep1 = dpi10_rep1[
        np.arange(dpi10_rep1.n_obs) != np.argmin(dpi10_rep1.obsm["spatial"][:, 0])
    ]
    dpi10_rep1 = dpi10_rep1[
        np.arange(dpi10_rep1.n_obs) != np.argmin(dpi10_rep1.obsm["spatial"][:, 0])
    ]
    print("loaded DPI10")

    dpi10_rep1 = dpi10_rep1[dpi10_rep1.obs["inj_uninj"] == "inj"]
    dpi10_rep1 = dpi10_rep1[dpi10_rep1.obs["D_V"] == "D"]

    dpi15_rep4 = sc.read_h5ad(
        "../../Data/ARISTA/15DPI_3.h5ad"
    )
    dpi15_rep4.obsm["spatial"] = dpi15_rep4.obsm["spatial"] * np.array([-1, -1])
    dpi15_rep4 = dpi15_rep4[
        np.arange(dpi15_rep4.n_obs) != np.argmax(dpi15_rep4.obsm["spatial"][:, 1])
    ]
    dpi15_rep4.X = dpi15_rep4.layers["counts"].copy()
    print("loaded DPI15")

    dpi15_rep4 = dpi15_rep4[dpi15_rep4.obs["inj_uninj"] == "inj"]
    dpi15_rep4 = dpi15_rep4[dpi15_rep4.obs["D_V"] == "D"]

    dpi20_rep2 = sc.read_h5ad(
        "../../Data/ARISTA/20DPI_2.h5ad"
    )
    dpi20_rep2.obsm["spatial"] = dpi20_rep2.obsm["spatial"] * np.array([-1, -1])
    dpi20_rep2.X = dpi20_rep2.layers["counts"].copy()
    dpi20_rep2 = dpi20_rep2[
        np.arange(dpi20_rep2.n_obs) != np.argmin(dpi20_rep2.obsm["spatial"][:, 0])
    ]
    dpi20_rep2 = dpi20_rep2[
        np.arange(dpi20_rep2.n_obs) != np.argmin(dpi20_rep2.obsm["spatial"][:, 0])
    ]
    print("loaded DPI20")

    dpi20_rep2 = dpi20_rep2[dpi20_rep2.obs["inj_uninj"] == "inj"]
    dpi20_rep2 = dpi20_rep2[dpi20_rep2.obs["D_V"] == "D"]

    dpi30 = sc.read_h5ad(
        "../../Data/ARISTA/30DPI.h5ad"
    )
    dpi30.obsm["spatial"] = dpi30.obsm["spatial"] * np.array([-1, -1])
    dpi30.X = dpi30.layers["counts"].copy()
    print("loaded DPI30")

    dpi30 = dpi30[dpi30.obs["inj_uninj"] == "inj"]
    dpi30 = dpi30[dpi30.obs["D_V"] == "D"]

    adata = sc.concat((dpi2_rep1, dpi5_rep1, dpi10_rep1, dpi15_rep4, dpi20_rep2, dpi30))
    adata.obs_names_make_unique()

    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.filter_genes(adata, min_counts=1)
    sc.pp.calculate_qc_metrics(adata, inplace=True)

    sc.pp.filter_cells(adata, min_genes=200)
    sc.pp.filter_cells(adata, max_genes=adata.obs["n_genes_by_counts"].quantile(0.999))
    sc.pp.filter_genes(adata, min_cells=3)

    adata.obs["x"] = adata.obsm["spatial"][:, 0]
    adata.obs["y"] = adata.obsm["spatial"][:, 1]

    adata.obs = adata.obs.rename(columns={"Annotation": "cellType"})


    adata_all = []
    for b in adata.obs["Batch"].unique():
        idx = adata.obs["Batch"] == b
        adata_temp = adata[idx].copy()
        adata_temp.obs["x"] = adata_temp.obsm["spatial"][:, 0]
        adata_temp.obs["y"] = adata_temp.obsm["spatial"][:, 1]
        adata_all.append(adata_temp)

    adata_sc = prepare_template(threshold=threshold)

    B = find_B_music(adata_sc)
    pd_B = B.T

    # genes_subset = select_informative_genes(pd_B, fc_thresh_ln=threshold)
    adata_sc.X = adata_sc.layers["counts"].copy()
    counts, cell_types = from_anndata(adata_sc, celltype_key="cellType")
    genes_subset = select_informative_genes(pd_B, counts, cell_types, fc_thresh_ln=threshold)
    pd_B = pd_B.loc[:, genes_subset]
    if return_sc == False:
        return adata_all, pd_B
    else:
        return adata_all, pd_B, adata_sc
    
# -----------------------------
# I/O helpers
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


def _parse_grid(spec: str, kind: str):
    a, b, n = spec.split(",")
    a, b, n = float(a), float(b), int(n)
    if kind == "log":
        if a <= 0 or b <= 0:
            raise ValueError("log grid requires positive a,b")
        return np.logspace(np.log10(a), np.log10(b), n)
    return np.linspace(a, b, n)


def baseline_out_dir(base_dir: Path, baseline_mu: float, lam: float) -> Path:
    return Path(base_dir) / f"mu_{baseline_mu:.3e}" / f"lam_{lam:.3e}"


def load_baseline_V(base_dir: Path, baseline_mu: float, lam: float):
    d = baseline_out_dir(base_dir, baseline_mu, lam)
    with open(d / "V.pkl", "rb") as f:
        return pickle.load(f)

def load_baseline_pi(base_dir: Path, baseline_mu: float, lam: float):
    d = baseline_out_dir(base_dir, baseline_mu, lam)
    with open(d / "pi.pkl", "rb") as f:
        return pickle.load(f)


def ensure_baseline(base_dir: Path, adata_all, pd_B, baseline_mu: float, lam: float, overwrite: bool, ot_solver, alpha):
    d = baseline_out_dir(base_dir, baseline_mu, lam)
    if (d / "_SUCCESS").exists() and not overwrite:
        print(f"[BASELINE] exists: {d}")
        return
    print(f"[BASELINE] computing mu={baseline_mu:.3e}, lam={lam:.3e}")
    V0, pi0, aux0 = solve_V_all_slices(
        adata_all, pd_B,
        lam=lam, mu=baseline_mu, time=None,
        outer_max=20, niter_max=int(1e3), tol=1e-5,
        eta=1e-3, verbose=False,
        ot_solver=ot_solver,
        one_step=True, alpha=alpha
    )
    save_ragged_results(d, V0, pi0, baseline_mu, lam, aux=aux0)


# -----------------------------
# Main: baseline first, then grid
# -----------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mu_grid", required=True)         # e.g. "1e4,1e6,3"
    p.add_argument("--lam_grid", required=True)        # e.g. "1e0,1e6,7"
    p.add_argument("--mu_grid_kind", choices=["log", "lin"], default="log")
    p.add_argument("--lam_grid_kind", choices=["log", "lin"], default="log")
    p.add_argument("--base_dir", default="results_arista")
    p.add_argument("--baseline_mu", type=float, default=0.0)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--overwrite_baseline", action="store_true")
    p.add_argument("--ot_solver_baseline", default="sinkhorn")
    p.add_argument("--ot_solver_main", default="fgw")
    p.add_argument("--alpha", type=float, default=0.9)
    args = p.parse_args()

    mus = _parse_grid(args.mu_grid, args.mu_grid_kind)
    lams = _parse_grid(args.lam_grid, args.lam_grid_kind)

    base_dir = Path(args.base_dir)
    baseline_mu = float(args.baseline_mu)

    try:
        print("[INFO] prepare_data() ...")
        adata_all, pd_B = prepare_data()

        # 1) Baselines
        print("[INFO] computing baselines ...")
        for lam in lams:
            ensure_baseline(
                base_dir=base_dir,
                adata_all=adata_all,
                pd_B=pd_B,
                baseline_mu=baseline_mu,
                lam=float(lam),
                overwrite=args.overwrite_baseline,
                ot_solver = args.ot_solver_main,
                alpha = args.alpha
            )

        # 2) Full grid
        print("[INFO] running grid ...")
        for mu in mus:
            for lam in lams:
                mu = float(mu)
                lam = float(lam)
                out_dir = base_dir / f"mu_{mu:.3e}" / f"lam_{lam:.3e}"
                if (out_dir / "_SUCCESS").exists() and not args.overwrite:
                    print(f"[SKIP] {out_dir}")
                    continue

                V_init = None
                pi_init = None
                if mu != baseline_mu:
                    V_init = load_baseline_V(base_dir, baseline_mu, lam)
                    pi_init = load_baseline_pi(base_dir, baseline_mu, lam)

                print(f"[RUN] mu={mu:.3e}, lam={lam:.3e}")
                if V_init is None:
                    V, pi, aux = solve_V_all_slices(
                        adata_all, pd_B,
                        lam=lam, mu=mu, time=None,
                        outer_max=10, niter_max=int(1e3), tol=1e-5,
                        eta=1e-3, verbose=False,
                        ot_solver=str(args.ot_solver_main),
                        one_step=True,
                        alpha = args.alpha
                    )
                    save_ragged_results(out_dir, V, pi, mu, lam, aux=aux)
                # if pi_init is None:
                    

                    
                else:
                    V, pi, aux = solve_V_all_slices(
                        adata_all, pd_B,
                        lam=lam, mu=mu, time=None,
                        outer_max=10, niter_max=int(1e3), tol=1e-5,
                        eta=1e-3, verbose=False,
                        V_init=V_init,
                        coupling = pi_init,
                        ot_solver=str(args.ot_solver_main),
                        one_step=True,
                        alpha = args.alpha
                    )

                    save_ragged_results(out_dir, V, None, mu, lam, aux=aux)

    except Exception as e:
        (base_dir / "_FAILED.txt").write_text(
            f"Error: {repr(e)}\n\nTraceback:\n{traceback.format_exc()}\n"
        )
        raise


if __name__ == "__main__":
    main()

