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


# -----------------------------
# Your data preparation (as-is)
# -----------------------------
def prepare_data(threshold=3.0,data_dir=None,return_sc=False):
    print(data_dir)
    if data_dir == None:
        data_dir = "../../"
    adata_scRNA3  = sc.read_h5ad(os.path.join(data_dir, "Data/ZESTA/zf3_scRNA.h5ad"))
    adata_scRNA3.X = adata_scRNA3.layers["counts"].copy()

    adata_scRNA5  = sc.read_h5ad(os.path.join(data_dir, "Data/ZESTA/zf5_scRNA.h5ad"))
    adata_scRNA5.X = adata_scRNA5.layers["counts"].copy()

    adata_scRNA10 = sc.read_h5ad(os.path.join(data_dir, "Data/ZESTA/zf10_scRNA.h5ad"))
    adata_scRNA10.X = adata_scRNA10.layers["counts"].copy()

    adata_scRNA12 = sc.read_h5ad(os.path.join(data_dir, "Data/ZESTA/zf12_scRNA.h5ad"))
    adata_scRNA12.X = adata_scRNA12.layers["counts"].copy()

    adata_scRNA18 = sc.read_h5ad(os.path.join(data_dir, "Data/ZESTA/zf18_scRNA.h5ad"))
    adata_scRNA18.X = adata_scRNA18.layers["counts"].copy()

    adata_scRNA24 = sc.read_h5ad(os.path.join(data_dir, "Data/ZESTA/zf24_scRNA.h5ad"))
    adata_scRNA24.X = adata_scRNA24.layers["counts"].copy()

    def fix_ann(adata):
        for k, v in adata.obsm.items():
            if hasattr(v, "values"):
                adata.obsm[k] = v.values
        for k, v in adata.varm.items():
            if hasattr(v, "values"):
                adata.varm[k] = v.values
        return adata

    adata_list = [adata_scRNA3, adata_scRNA5, adata_scRNA10, adata_scRNA12, adata_scRNA18, adata_scRNA24]
    adata_list = [fix_ann(a) for a in adata_list]
    adata_scRNA = sc.concat(adata_list)
    adata_scRNA.obs_names_make_unique()

    import re
    def coarse_celltype(x: str) -> str:
        x = re.sub(r"\s+", " ", str(x).strip())
        if x == "UND" or "Hsp" in x or "Apoptotic" in x:
            return "QC / unknown"
        if ("Leukocyte" in x) or ("Erythroid" in x) or re.search(r"Endothelial|Blood Vessel Endothelial|Cardiovascular", x):
            return "Vascular / blood / immune"
        if re.search(r"Otic|Optic|Lens|Pronephros|Hatching", x):
            return "Sensory / organs"
        if re.search(r"EVL|Periderm|Integument|Epidermis|YSL", x):
            return "Epidermis / Periderm / EVL / YSL"
        if re.search(r"Neural Stem Cells|Primary Neuron|GABAergic Neuron|Neuron|Neural Plate|Neural Keel|Neural Rod|Central Nervous System|Brain|Spinal Cord", x):
            return "Neural / CNS"
        if re.search(r"Neural Crest|Pigment", x):
            return "Neural crest / pigment"
        if "Blastodisc" in x:
            return "Early embryo / blastoderm"
        if re.search(r"Notochord|Floor Plate", x):
            return "Midline (notochord/floor plate)"
        if re.search(r"Paraxial|Segmental Plate|Somite|Tail bud|Tail Bud|Tailbud", x):
            return "Mesoderm (paraxial/somitic/tailbud)"
        if "Lateral Plate Mesoderm" in x:
            return "Mesoderm (LPM)"
        if re.search(r"Endoderm|Pharynx|Pharyngeal", x):
            return "Endoderm / pharynx"
        if "Presumptive" in x or "," in x or "Dorsal Margin" in x or "Polster" in x:
            return "Presumptive germ layers"
        return "QC / unknown"

    adata_scRNA.obs["celltype_coarse"] = (
        adata_scRNA.obs["celltype_new"].map(coarse_celltype).astype("category")
    )

    sc.pp.normalize_total(adata_scRNA, target_sum=1e4)
    sc.pp.filter_genes(adata_scRNA, min_counts=1)

    del adata_scRNA3, adata_scRNA5, adata_scRNA10, adata_scRNA12, adata_scRNA18, adata_scRNA24

    zf3 = sc.read_h5ad(os.path.join(data_dir, "Data/ZESTA/zf3_stereoseq.h5ad"))
    zf3 = zf3[zf3.obs["slice"] == 1]
    zf3.X = zf3.layers["counts"].copy()

    zf5 = sc.read_h5ad(os.path.join(data_dir, "Data/ZESTA/zf5_stereoseq.h5ad"))
    zf5 = zf5[zf5.obs["slice"] == 10]
    zf5.X = zf5.layers["counts"].copy()

    zf10 = sc.read_h5ad(os.path.join(data_dir, "Data/ZESTA/zf10_stereoseq.h5ad"))
    zf10 = zf10[(zf10.obs["slice"] == 11) | (zf10.obs["slice"] == 17)]
    zf10.X = zf10.layers["counts"].copy()

    zf12 = sc.read_h5ad(os.path.join(data_dir, "Data/ZESTA/zf12_stereoseq.h5ad"))
    zf12 = zf12[(zf12.obs["slice"] == 8) | (zf12.obs["slice"] == 5)]
    zf12.X = zf12.layers["counts"].copy()

    zf18 = sc.read_h5ad(os.path.join(data_dir, "Data/ZESTA/zf18_stereoseq.h5ad"))
    zf18 = zf18[(zf18.obs["slice"] == 8) | (zf18.obs["slice"] == 11)]
    zf18.X = zf18.layers["counts"].copy()

    zf24 = sc.read_h5ad(os.path.join(data_dir, "Data/ZESTA/zf24_stereoseq.h5ad"))
    zf24 = zf24[zf24.obs["slice"] == 4]
    zf24.X = zf24.layers["counts"].copy()

    adata = sc.concat((zf3, zf5, zf10, zf12, zf18, zf24))
    adata.obs_names_make_unique()
    del zf3, zf5, zf10, zf12, zf18, zf24

    adata.obsm["spatial"] = adata.obs[["spatial_x", "spatial_y"]].values
    adata.obs["Batch"] = adata.obs["time_point"].astype(str) + " slice " + adata.obs["slice"].astype(str)

    adata.obsm["spatial"][adata.obs["Batch"] == "18hpf slice 8"]  = adata.obsm["spatial"][adata.obs["Batch"] == "18hpf slice 8"]  @ np.array([[0, 1], [1, 0]])
    adata.obsm["spatial"][adata.obs["Batch"] == "10hpf slice 17"] = adata.obsm["spatial"][adata.obs["Batch"] == "10hpf slice 17"] @ np.array([[0, 1], [1, 0]])

    theta = -7 * np.pi / 4
    adata.obsm["spatial"][adata.obs["Batch"] == "12hpf slice 5"] *= [1, -1]
    adata.obsm["spatial"][adata.obs["Batch"] == "12hpf slice 5"] = adata.obsm["spatial"][adata.obs["Batch"] == "12hpf slice 5"] @ np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])

    theta = -np.pi / 8
    adata.obsm["spatial"][adata.obs["Batch"] == "12hpf slice 8"] = adata.obsm["spatial"][adata.obs["Batch"] == "12hpf slice 8"] @ np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    adata.obsm["spatial"][adata.obs["Batch"] == "10hpf slice 11"] *= [-1, 1]
    adata.obsm["spatial"][adata.obs["Batch"] == "12hpf slice 8"] *= [-1, 1]
    adata.obsm["spatial"][adata.obs["Batch"] == "24hpf slice 4"] *= [-1, 1]
    adata.obsm["spatial"][adata.obs["Batch"] == "18hpf slice 11"] *= [1, -1]

    adata.obs["x"] = adata.obsm["spatial"][:, 0]
    adata.obs["y"] = adata.obsm["spatial"][:, 1]
    adata_scRNA.obs = adata_scRNA.obs.rename(columns={"celltype_new": "cellType"})

    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.filter_genes(adata, min_counts=1)

    adata_all = []
    for b in adata.obs["Batch"].unique():
        idx = adata.obs["Batch"] == b
        adata_temp = adata[idx].copy()
        adata_temp.obs["x"] = adata_temp.obsm["spatial"][:, 0]
        adata_temp.obs["y"] = adata_temp.obsm["spatial"][:, 1]
        adata_all.append(adata_temp)

    slices = [0, 1, 2, 5, 6, 8]
    adata_insamp = [adata_all[i] for i in slices]

    B = find_B_music(adata_scRNA)
    pd_B = B.T
    counts, cell_types = from_anndata(adata_scRNA, celltype_key="cellType")
    adata_scRNA.X = adata_scRNA.layers["counts"].copy()
    genes_subset = select_informative_genes(pd_B, counts, cell_types, fc_thresh_ln=threshold)#, dispersion_quantile=0.99)
    pd_B = pd_B.loc[:, genes_subset]
    if return_sc == False:
        return adata_insamp, pd_B
    else:
        return adata_insamp, pd_B, adata_scRNA


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
        ot_solver=ot_solver, alpha=alpha
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
    p.add_argument("--base_dir", default="results_zesta")
    p.add_argument("--baseline_mu", type=float, default=0.0)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--overwrite_baseline", action="store_true")
    p.add_argument("--threshold", type=float, default=1.25)
    # p.add_argument("--ot_solver_baseline", default="sinkhorn")
    p.add_argument("--ot_solver_main", default="fgw")
    p.add_argument("--alpha",type=float, default=0.9)
    args = p.parse_args()

    mus = _parse_grid(args.mu_grid, args.mu_grid_kind)
    lams = _parse_grid(args.lam_grid, args.lam_grid_kind)

    base_dir = Path(args.base_dir)
    baseline_mu = float(args.baseline_mu)

    try:
        print("[INFO] prepare_data() ...")
        adata_all, pd_B = prepare_data(threshold = args.threshold)

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
                        alpha = args.alpha
                    )
                    save_ragged_results(out_dir, V, pi, mu, lam, aux=aux)
                else:
                    V, pi, aux = solve_V_all_slices(
                        adata_all, pd_B,
                        lam=lam, mu=mu, time=None,
                        outer_max=10, niter_max=int(1e3), tol=1e-5,
                        eta=1e-3, verbose=False,
                        V_init=V_init,
                        coupling=pi_init,
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

