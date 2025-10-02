import numpy as np
import pandas as pd
from scipy.sparse import issparse, csc_matrix

def createscRef_py(adata_sc, ct_select=None, ct_varname="cellType", sample_varname=None):
    """
    Build CARD-style reference basis B (genes x celltypes) from single-cell AnnData.
    Expects adata_sc.X = cells x genes (raw counts).
    """

    def _as2d(a):
        a = np.asarray(a)
        return a if a.ndim == 2 else a.reshape(-1, 1)

    # ----- labels & inputs -----
    genes = pd.Index(adata_sc.var_names)
    ct = adata_sc.obs[ct_varname].astype(str)
    if ct_select is None:
        ct_select = list(ct.dropna().unique())
    if sample_varname is None:
        sample = pd.Series(["Sample"] * adata_sc.n_obs, index=adata_sc.obs_names)
    else:
        sample = adata_sc.obs[sample_varname].astype(str)

    # counts: genes x cells
    X_cxg = adata_sc.X
    X_gxc = X_cxg.T if issparse(X_cxg) else np.asarray(X_cxg, dtype=float).T  # (g x c)

    # ----- library sizes per (ct, sample) -> S -----
    if issparse(X_cxg):
        lib_sizes = np.asarray(X_cxg.sum(axis=1)).ravel()
    else:
        lib_sizes = np.asarray(X_cxg.sum(axis=1)).ravel()

    df_sum = (
        pd.DataFrame({"ct": ct.values, "sample": sample.values, "sum": lib_sizes})
          .groupby(["ct", "sample"], as_index=False)["sum"].sum()
    )
    wide_sum = df_sum.pivot(index="sample", columns="ct", values="sum").fillna(0.0)

    tbl = pd.crosstab(sample, ct)                           # cell counts per (sample, ct)
    wide_sum = wide_sum.reindex(index=tbl.index, columns=tbl.columns).fillna(0.0)

    with np.errstate(divide="ignore", invalid="ignore"):
        S_JK = wide_sum.values / np.where(tbl.values == 0, np.nan, tbl.values)
    S = np.nanmean(S_JK, axis=0)                            # order = wide_sum.columns

    # ----- per (gene, ct$*$sample) means -> Theta_S_rowMean -----
    ct_sample = pd.Categorical([f"{c}$*${s}" for c, s in zip(ct.values, sample.values)])
    G = pd.get_dummies(ct_sample, sparse=True)              # (cells x groups)
    groups = list(G.columns)

    # sums: (genes x cells) @ (cells x groups) = (genes x groups)
    if issparse(X_gxc):
        S_gxg = X_gxc @ csc_matrix(G.values)
        S_gxg = S_gxg.toarray()                             # ensure ndarray
    else:
        S_gxg = X_gxc @ G.values
    S_gxg = _as2d(S_gxg)

    n_per_group = np.asarray(G.sum(axis=0)).ravel().astype(float)
    n_per_group[n_per_group == 0] = 1.0

    Theta_S_rowMean = _as2d(S_gxg / n_per_group)            # (genes x groups)
    Theta_S_rowSums = _as2d(Theta_S_rowMean * n_per_group)  # scale back by sizes
    col_sums = Theta_S_rowSums.sum(axis=0, keepdims=True)
    col_sums[col_sums == 0] = 1.0
    Theta_S = _as2d(Theta_S_rowSums / col_sums)             # normalized per group

    # ----- average across samples within each cell type -> Theta (genes x ct) -----
    pure_ct = [g.split("$*$")[0] for g in groups]
    ct_unique = list(pd.Series(ct.values).dropna().unique())

    Theta_blocks = []
    ct_final = []
    for ctype in ct_unique:
        idx = np.asarray([i for i, pc in enumerate(pure_ct) if pc == ctype], dtype=int)
        if idx.size == 0:
            continue
        block = Theta_S[:, idx]                             # stays 2D even if one column
        Theta_blocks.append(block.mean(axis=1, keepdims=True))
        ct_final.append(ctype)

    if not Theta_blocks:
        raise ValueError("No cell types found for basis construction.")

    Theta = np.hstack(Theta_blocks)                         # genes x len(ct_final)

    # ----- match S to Theta’s column order and scale: basis = Theta * S -----
    S_series = pd.Series(S, index=wide_sum.columns)         # align S to ct columns
    S_vec = S_series.reindex(ct_final).fillna(0.0).values
    basis = Theta * S_vec                                   # columnwise scaling

    # keep only requested ct_select, in that order
    keep_ct = [c for c in ct_select if c in ct_final]
    if not keep_ct:
        raise ValueError("None of ct_select found in data.")
    col_idx = [ct_final.index(c) for c in keep_ct]

    B = pd.DataFrame(basis[:, col_idx], index=genes, columns=keep_ct)
    return {"basis": B}

