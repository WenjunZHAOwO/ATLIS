import numpy as np
import scipy
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Wedge, Patch
import scanpy as sc
from scipy.sparse import lil_matrix, csr_matrix
import ot
from ot.gromov import gromov_wasserstein, fused_gromov_wasserstein

import numpy as np
import scipy.sparse as sp

import sys
import torch
import sys
import FRLC

from ott.geometry.pointcloud import PointCloud
from ott.problems.linear.linear_problem import LinearProblem
# from ott.problems.quadratic.quadratic_problem import QuadraticProblem
from ott.solvers.linear.sinkhorn import Sinkhorn
# from ott.solvers.quadratic.gromov_wasserstein import GromovWasserstein


def _normalize_cols(M, eps=1e-12):
    """
    Column-normalize so each column sums to 1.
    Works for dense numpy arrays and scipy.sparse (CSR/CSC/COO).
    """
    if sp.issparse(M):
        # Convert to CSC for efficient column ops
        M = M.tocsc(copy=False)
        col_sums = np.asarray(M.sum(axis=0)).ravel()   # shape (n,)
        # avoid divide-by-zero
        col_sums[col_sums <= eps] = 1.0
        inv = sp.diags(1.0 / col_sums)                # (n,n)
        return M @ inv                                 # still sparse (CSC)
    else:
        M = np.asarray(M, dtype=float, order='C')      # ensure numeric dense
        col_sums = M.sum(axis=0, keepdims=True)        # (1,n)
        col_sums[col_sums <= eps] = 1.0
        return M / col_sums



    
def assemble_blocks(diag_blocks, offdiag_blocks=()):
    """
    diag_blocks: list of 2D arrays (dense or sparse), block i has shape (ri, ci)
    offdiag_blocks: iterable of (i, j, Bij) where Bij has shape (ri, cj)

    Returns: CSR sparse matrix with the given diagonal and off-diagonal blocks.
    """
    # sizes per block row/column
    r_sizes = [b.shape[0] for b in diag_blocks]
    c_sizes = [b.shape[1] for b in diag_blocks]

    # total shape
    n_rows = sum(r_sizes)
    n_cols = sum(c_sizes)
    A = lil_matrix((n_rows, n_cols))

    # offsets
    r_off = np.r_[0, np.cumsum(r_sizes[:-1])]
    c_off = np.r_[0, np.cumsum(c_sizes[:-1])]

    # place diagonal blocks
    for i, Bii in enumerate(diag_blocks):
        r0, c0 = r_off[i], c_off[i]
        ri, ci = Bii.shape
        A[r0:r0+ri, c0:c0+ci] = Bii

    # place off-diagonal blocks
    for (i, j, Bij) in offdiag_blocks:
        ri, cj = r_sizes[i], c_sizes[j]
        if Bij.shape != (ri, cj):
            raise ValueError(f"Offdiag block ({i},{j}) must be shape {(ri, cj)}, got {Bij.shape}")
        r0, c0 = r_off[i], c_off[j]
        A[r0:r0+ri, c0:c0+cj] = Bij

    return A.tocsr()


def find_B(adata):
    # Given data_ref with single-cell,
    # find average expression level for all possible cell types
    # Convert sparse to dense if needed (but stay careful with memory)
    # sc.pp.log1p(adata)
    sc.pp.filter_genes(adata, min_counts=1)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.filter_genes(adata, min_counts=1)
    X = adata.X
    if not isinstance(X, np.ndarray):
        X = X.toarray()   # optional; you can also work with sparse

    # Create a DataFrame: rows=cells, cols=genes
    expr_df = pd.DataFrame(X, index=adata.obs.index, columns=adata.var_names)

    # Add cell type column
    expr_df['cellType'] = adata.obs['cellType'].values

    # Group by cell type and compute mean expression across cells
    avg_expr = expr_df.groupby('cellType').mean()
    avg_expr *= 5
    return avg_expr 

import numpy as np
import pandas as pd
from typing import Optional, Tuple, Dict, Any
from scipy import sparse

import numpy as np
import pandas as pd

import re
import numpy as np

# helper
_MONTHS = "(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)"
_DATE_LIKE_RE = re.compile(rf"^\d{{1,2}}-{_MONTHS}$")

import numpy as np
import pandas as pd

import pandas as pd
import scipy.sparse as sp

def from_anndata(adata_scRNA, celltype_key="cellType", layer_key="counts"):
    X = adata_scRNA.layers[layer_key] if layer_key in adata_scRNA.layers else adata_scRNA.X
    if sp.issparse(X):
        X = X.toarray()

    counts = pd.DataFrame(
        X.T, index=adata_scRNA.var_names, columns=adata_scRNA.obs_names
    )
    cell_types = adata_scRNA.obs[celltype_key].astype(str).to_numpy()
    return counts, cell_types


def select_informative_genes(
    B,
    counts,
    cell_types,
    common_genes=None,
    fc_thresh_ln=1.25,
    dispersion_quantile=1.0,
    drop_date_like=True,
    cell_type_key=None
):
    """
    Parameters
    ----------
    B : pd.DataFrame
        Cell types × genes (rows = cell types, columns = genes)
    counts : pd.DataFrame
        Genes × cells raw count matrix
    cell_types : array-like
        Cell-type label per cell (length = n_cells)
    common_genes : set or list, optional
        Restrict genes to this set (R: commonGene)
    fc_thresh_ln : float
        Natural-log fold-change threshold (R uses 1.25)
    dispersion_quantile : float
        Quantile cutoff for removing highly overdispersed genes (R uses 0.99)
    """

    eps = 1e-6
    gene_set = set()

    # ---------- Step 1: fold-change filter (matches R) ----------
    for ict in B.index:
        rest = B.drop(index=ict).mean(axis=0)  # mean over other cell types
        FC = np.log(B.loc[ict] + eps) - np.log(rest + eps)
        keep = (FC > fc_thresh_ln) & (B.loc[ict] > 0)
        gene_set.update(B.columns[keep])

    genes = sorted(gene_set)

    if common_genes is not None:
        genes = [g for g in genes if g in common_genes]

    if len(genes) == 0:
        return []

    # ---------- Step 2: dispersion filter (variance / mean) ----------
    counts_sub = counts.loc[counts.index.intersection(genes)]

    # only keep cell types with >= 2 cells
    ct_series = pd.Series(cell_types, index=counts.columns)
    valid_cts = ct_series.value_counts()
    valid_cts = valid_cts[valid_cts > 1].index

    disp = []

    for ct in valid_cts:
        cols = ct_series[ct_series == ct].index
        X = counts_sub[cols]

        mean = X.mean(axis=1)
        var = X.var(axis=1)

        disp.append(var / (mean + eps))

    # genes × cell types dispersion matrix
    disp = pd.concat(disp, axis=1)

    mean_disp = disp.mean(axis=1, skipna=True)
    if dispersion_quantile < 1:
        cutoff = mean_disp.quantile(dispersion_quantile)

        genes_final = mean_disp[mean_disp < cutoff].index.tolist()
    else:
        genes_final = genes

    if drop_date_like:
        genes_final = [
            g for g in genes_final
            if not _DATE_LIKE_RE.fullmatch(str(g))
        ]

    return sorted(genes_final)




import numpy as np
import pandas as pd

def select_informative_genes_pairwise(B, adata=None, fc_thresh_log2=1.25,
                             drop_date_like=True, quantile_cut=0.99) -> list:
    eps = 1e-6
    fc_thresh_ln = fc_thresh_log2 * np.log(2)  # convert log2 FC → ln

    gene_set = set()
    for ict in B.index:
        rest = B.drop(index=ict).mean(axis=0)
        FC = np.log(B.loc[ict] + eps) - np.log(rest + eps)
        keep = (FC > fc_thresh_ln) & (B.loc[ict] > 0)
        gene_set.update(B.columns[keep])

    genes = sorted(gene_set)

    # optional: drop date-like gene names
    if drop_date_like:
        genes = [g for g in genes if not _DATE_LIKE_RE.fullmatch(str(g))]

    # -------- add CARD-style dispersion filter --------
    if adata is not None and "celltype" in adata.obs.columns:
        # get scRNA counts (cells × genes) as DataFrame
        sc_df = adata.to_df().T    # genes × cells
        genes = [g for g in genes if g in sc_df.index]
        if genes:
            sc_sub = sc_df.loc[genes]
            celltypes = adata.obs["celltype_new"]
            disp_list = []
            for ct, idx in celltypes.groupby(celltypes).groups.items():
                idx = list(idx)
                if len(idx) < 2:
                    continue
                Xct = sc_sub[idx]
                var = Xct.var(axis=1)
                mean = Xct.mean(axis=1) + eps
                disp_list.append(var / mean)
            if disp_list:
                disp_mean = pd.concat(disp_list, axis=1).mean(axis=1)
                cutoff = disp_mean.quantile(quantile_cut)
                genes = disp_mean[disp_mean < cutoff].index.tolist()
    # --------------------------------------------------

    return genes












def find_B_music(
    adata,
    ct_varname: str = "cellType",
    sample_varname: Optional[str] = "sampleInfo",
    return_parts: bool = False,
    eps: float = 1e-12
):# -> (pd.DataFrame | Tuple[pd.DataFrame, Dict[str, Any]]):
    """
    Build MuSiC-style reference:
        Theta_{gk} = m_k * mu_{gk}
    where:
        mu_{gk} = mean_donor( mu_{gk}^{(d)} ),   sum_g mu_{gk} = 1 for each k
        m_k     = mean_donor( m_k^{(d)} ),       average total RNA per cell of type k

    Inputs
    ------
    adata: AnnData with
        - X: counts (raw preferred; avoid log1p for this step)
        - obs[ct_varname]: cell type labels
        - obs[sample_varname] (optional): donor / subject IDs
    ct_varname: column in adata.obs with cell type labels
    sample_varname: column in adata.obs with donor IDs; if None or missing,
                    all cells are treated as coming from one “donor”
    return_parts: if True, also return dict with mu, m, and per-donor pieces
    eps: small constant to avoid divide-by-zero

    Returns
    -------
    Theta_df: DataFrame (genes x cellTypes)
    If return_parts=True, also returns a dict with:
        - mu_df: genes x cellTypes (averaged across donors)
        - m_series: cellTypes (averaged across donors)
        - mu_d: nested dict {donor: DataFrame genes x cellTypes}
        - m_d: nested dict {donor: Series cellTypes}
    """
    # --- pull matrix safely (works for dense or sparse) ---
    # sc.pp.filter_genes(adata, min_counts=1)
    # sc.pp.normalize_total(adata, target_sum=1e2)
    # sc.pp.filter_genes(adata, min_counts=1)

    
    X = adata.X
    if sparse.issparse(X):
        X = X.tocsr()
    elif not isinstance(X, np.ndarray):
        X = np.asarray(X)

    genes = np.array(adata.var_names)
    cell_types = np.array(adata.obs[ct_varname])

    # optional donors
    if (sample_varname is not None) and (sample_varname in adata.obs):
        donors = np.array(adata.obs[sample_varname])
    else:
        donors = np.array(["_ONE_DONOR_"] * adata.n_obs)

    # --- containers for per-donor mu and m ---
    donors_unique = pd.unique(donors)
    cts_unique = pd.unique(cell_types)

    mu_d: Dict[str, pd.DataFrame] = {}   # genes x cellTypes for each donor
    m_d: Dict[str, pd.Series] = {}       # cellTypes for each donor
    
    # --- compute per-donor, per-cell-type summaries ---
    # We’ll do this donor-by-donor to keep memory tame on big datasets.
    for d in donors_unique:
        mask_d = (donors == d)
        if sparse.issparse(X):
            X_d = X[mask_d, :]
        else:
            X_d = X[mask_d, :]

        ct_d = cell_types[mask_d]
        # per-cell library size (total counts)
        if sparse.issparse(X_d):
            libsizes = np.array(X_d.sum(axis=1)).ravel()
        else:
            libsizes = X_d.sum(axis=1)

        # Initialize accumulators for this donor
        mu_cols = []
        m_vals = []

        for k in cts_unique:
            mask_k = (ct_d == k)
            if not np.any(mask_k):
                # donor d has zero cells of this type -> NaNs; handle later
                mu_cols.append(pd.Series(np.nan, index=genes, name=k))
                m_vals.append(np.nan)
                continue

            if sparse.issparse(X_d):
                X_dk = X_d[mask_k, :]
            else:
                X_dk = X_d[mask_k, :]

            # mean expression per gene for donor d, type k
            if sparse.issparse(X_dk):
                mean_counts = np.array(X_dk.mean(axis=0)).ravel()
            else:
                mean_counts = X_dk.mean(axis=0)

            # convert to relative expression within cell type k
            total = mean_counts.sum()
            if total <= eps:
                mu_gk = np.zeros_like(mean_counts, dtype=float)
            else:
                mu_gk = mean_counts / (total + eps)

            # cell size: average total counts per cell (library size) for this donor & type
            m_kd = np.mean(libsizes[mask_k])

            mu_cols.append(pd.Series(mu_gk, index=genes, name=k))
            m_vals.append(m_kd)

        mu_d[d] = pd.concat(mu_cols, axis=1)   # genes x cellTypes
        m_d[d] = pd.Series(m_vals, index=cts_unique, name=d)

    # --- average across donors (ignoring NaNs where donor lacks a type) ---
    # mu: gene-wise mean across donors per cell type
    mu_stack = []
    for d in donors_unique:
        mu_stack.append(mu_d[d].reindex(index=genes, columns=cts_unique))
    mu_3d = np.stack([df.values for df in mu_stack], axis=2)  # shape: G x K x D

    # average across donors with NaN handling
    mu_df = pd.DataFrame(
        np.nanmean(mu_3d, axis=2),
        index=genes,
        columns=cts_unique
    )
    # renormalize columns (just in case NaNs/eps caused slight drift)
    mu_df = mu_df.div(mu_df.sum(axis=0).replace(0, np.nan), axis=1).fillna(0.0)

    # m: mean across donors per cell type
    m_mat = pd.concat(m_d, axis=1)  # rows: cellTypes, cols: donors (MultiIndex)
    m_series = m_mat.mean(axis=1, skipna=True)  # average across donors -> per cell type
    m_series = m_series.reindex(cts_unique).fillna(0.0)

    # --- Theta = m_k * mu_{gk} ---
    Theta = mu_df.copy()
    Theta = Theta * m_series.values  # broadcasts across rows (genes)

    Theta_df = Theta  # genes x cellTypes

    if return_parts:
        parts = dict(mu_df=mu_df, m_series=m_series, mu_d=mu_d, m_d=m_d)
        return Theta_df, parts
    else:
        return Theta_df


import numpy as np
from sklearn.metrics import pairwise_distances
from scipy.sparse import csgraph

import numpy as np
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix, issparse
from scipy.sparse.csgraph import laplacian as csgraph_laplacian

def laplacian_from_coords(x, y, k=20, mutual=False, normed=False, return_dense=False):
    """
    k-NN graph Laplacian from 2D coords (no Gaussian kernel).

    Parameters
    ----------
    x, y : array-like (n,)
    k : int                  number of neighbors
    mutual : bool            keep edge i<->j only if both are in each other's kNN
    normed : bool            symmetric normalized Laplacian if True; else unnormalized
    return_dense : bool      return dense np.ndarray instead of sparse

    Returns
    -------
    L : (n,n) sparse CSR (or dense) Laplacian
    W : (n,n) sparse CSR (or dense) adjacency with 1s on edges
    """
    coords = np.column_stack([x, y])
    n = coords.shape[0]
    k = min(k, n-1)

    # kNN (exclude self)
    nn = NearestNeighbors(n_neighbors=k+1, algorithm='auto').fit(coords)
    _, nbrs = nn.kneighbors(coords)         # (n, k+1)
    nbrs = nbrs[:, 1:]                      # drop self

    # Build directed edges i -> nbrs[i, :]
    row = np.repeat(np.arange(n), k)
    col = nbrs.ravel()
    data = np.ones_like(col, dtype=float)
    W_dir = csr_matrix((data, (row, col)), shape=(n, n))

    # Symmetrize
    if mutual:
        # keep only mutual edges
        W = W_dir.minimum(W_dir.T)
    else:
        # union (undirected)
        W = W_dir.maximum(W_dir.T)

    # Laplacian
    L = csgraph_laplacian(W, normed=normed)

    if return_dense:
        return L.toarray(), W.toarray()
    return L, W










def solve_V_per_slice(adata, pd_B, lam=10, k=20, n_top=2000, niter_max=1e3, tol=1e-5, eta = 1e-3, verbose=True, return_obj=False, proj_simplex=False, step_rate='backtrack'):
    # stop
    
    gene_names = adata.var_names
    common = gene_names.intersection(pd_B.columns)
        # print(common)
        # Subset AnnData and B, in the same order
    adata_sub = adata[:, common].copy()
    pd_B_sub = pd_B[common]
        
        
    x = adata_sub.obs['x'].to_numpy()
    y = adata_sub.obs['y'].to_numpy()

    L, W = laplacian_from_coords(x, y, k=k, normed=True)
    L = lam*L
    B = pd_B_sub.to_numpy().T
    # X = adata_sub.X.T
    X = adata_sub.layers['counts'].T if 'counts' in adata_sub.layers else adata_sub.X.T
    return solve_V_universal(X,B,L,niter_max=niter_max, tol=tol, eta = eta, verbose=verbose, return_obj=return_obj, proj_simplex=proj_simplex,step_rate = step_rate)
import numpy as np
from scipy import sparse

def solve_V_universal(
    X, B, L,
    niter_max=1e3, tol=1e-5, eta=1e-3, verbose=True,
    V0=None, return_obj=False, eps_norm=1e-12,
    proj_simplex=False, step_rate='backtrack'
):
    # Ensure sparse formats (cheap if already sparse)
    X = X.tocsr() if sparse.issparse(X) else np.asarray(X)
    B = B.tocsr() if sparse.issparse(B) else np.asarray(B)
    L = L.tocsr() if sparse.issparse(L) else np.asarray(L)

    # Shapes
    d, n = X.shape
    dB, r = B.shape
    assert d == dB, f"X is {X.shape}, B is {B.shape} (dim mismatch)"

    # Precompute Gram terms
    # G = X^T B  (n x r)
    # H = B^T B  (r x r)
    # For sparse, these are fine; convert to dense arrays for fast dense ops.
    G = (X.T @ B)
    H = (B.T @ B)

    if sparse.issparse(G): G = G.toarray()
    if sparse.issparse(H): H = H.toarray()

    # Precompute ||X||_F^2 once (for exact objective)
    if sparse.issparse(X):
        xnorm2 = float(X.multiply(X).sum())
    else:
        xnorm2 = float(np.sum(X * X))

    rng = np.random.default_rng(0)

    # Initialize V (n x r), dense
    if V0 is None:
        V = rng.random((n, r), dtype=float)
    else:
        V = np.asarray(V0, dtype=float)
        assert V.shape == (n, r)

    # ----- Sparse-safe objective (no residual matrix) -----
    # ||X - B V^T||_F^2 = ||X||_F^2 - 2 tr(V^T G) + tr(V^T V H)
    # regularizer = tr(V^T L V) = sum_ij V_ij * (L V)_ij
    def objective(V):
        # data term
        VH = V @ H                          # (n x r), dense
        data = xnorm2 - 2.0 * np.sum(V * G) + np.sum(VH * V)

        # reg term
        LV = L @ V                          # sparse@dense -> dense ndarray
        reg = np.sum(LV * V)

        return data + reg

    # Gradient: 2(VH - G) + 2(LV)
    def deriv_objective(V):
        return 2.0 * (V @ H - G) + 2.0 * (L @ V)

    f_prev = objective(V)

    for it in range(int(niter_max)):
        g = deriv_objective(V)
        V_old = V

        if step_rate == 'backtrack':
            step = float(eta)

            # Precompute ||g||_F^2 once for Armijo
            gnorm2 = float(np.sum(g * g))

            while True:
                V_trial = proj_simplex_rows(V_old - step * g, z=1.0, proj=proj_simplex)
                f_trial = objective(V_trial)

                if f_trial <= f_prev - 1e-4 * step * gnorm2:
                    V = V_trial
                    f_curr = f_trial
                    break

                step *= 0.5
                if step < 1e-8:
                    V = V_trial
                    f_curr = f_trial
                    break

        elif step_rate == 'automatic':
            # This heuristic assumes L is diagonal-dominant; use diagonal only
            dL = L.diagonal() if sparse.issparse(L) else np.diag(L)
            DV = dL[:, None] * V_old

            step_mat = V_old / (2.0 * (V_old @ H + DV + 1e-15))
            V = V_old - step_mat * g
            f_curr = objective(V)
            step = float(np.mean(step_mat))
        else:
            raise ValueError("step_rate must be 'backtrack' or 'automatic'")

        rel_drop = (f_prev - f_curr) / max(1.0, abs(f_prev))

        if verbose and (it % 50 == 0 or rel_drop < tol):
            if step_rate == 'backtrack':
                print(f"iter {it:4d}  f={f_curr:.6e}  rel_drop={rel_drop:.3e}  step={step:.2e}")
            else:
                print(f"iter {it:4d}  f={f_curr:.6e}  rel_drop={rel_drop:.3e}  mean step={step:.2e}")

        if f_curr <= f_prev and rel_drop < tol:
            if verbose:
                print("stopped after iteration #", it)
            break

        f_prev = f_curr

    if not return_obj:
        return V
    else:
        # Return exact split (still no residual formation)
        VH = V @ H
        data_term = xnorm2 - 2.0 * np.sum(V * G) + np.sum(VH * V)
        reg_term = np.sum((L @ V) * V)
        return V, reg_term, data_term

        




import anndata as ad
import ot
import numpy as np
import scipy.sparse as sp
import scipy.spatial

def solve_V_all_slices(
    adata_group, pd_B,
    lam=10, mu=1, time=None,
    outer_max=5, niter_max=1e3, tol=1e-5, eta=1e-3,
    verbose=True, coupling=None, k=10, init="per_slice",
    V_init=None, ot_solver="fgw", one_step=True,
    step_rate="automatic", alpha=0.9,
    dtype=np.float32,
):
    """
    Sparse-safe rewrite: never materializes R = X - B V^T.
    Assumes:
      - adata_group[i].X can be sparse (recommended).
      - laplacian_from_coords returns sparse (recommended).
      - assemble_blocks returns a scipy.sparse matrix.
      - solve_V_universal is the SPARSE-SAFE version (no residual build).
    Notes:
      - V is dense by design.
      - OT coupling still forms dense C and dense pi (can dominate memory).
    """

    rng = np.random.default_rng(0)
    n_times = len(adata_group)

    # offsets into stacked V
    n_spots = [0]
    for i in range(n_times):
        n_spots.append(n_spots[-1] + adata_group[i].shape[0])

    # concatenate and intersect genes
    adata = ad.concat(adata_group, axis=0, join="outer")
    gene_names = adata.var_names
    common = gene_names.intersection(pd_B.columns)

    adata_sub = adata[:, common].copy()
    for i in range(n_times):
        adata_group[i] = adata_group[i][:, common].copy()
    pd_B_sub = pd_B[common]

    # time rescaling
    if time is None:
        time = list(range(n_times))
    if time[-1] == time[0]:
        rescaled_time = [0.0 for _ in time]
    else:
        rescaled_time = [(t - time[0]) / (time[-1] - time[0]) for t in time]

    # build spatial Laplacians (diag blocks)
    diag_blocks = [None] * n_times
    for i in range(n_times):
        x = adata_group[i].obs["x"].to_numpy()
        y = adata_group[i].obs["y"].to_numpy()
        Li, _ = laplacian_from_coords(x, y, k=k, normed=True)  # should be sparse
        if not sp.issparse(Li):
            Li = sp.csr_matrix(Li)
        diag_blocks[i] = (lam * Li).tocsr()

    # B: genes x r (dense ok), X: genes x N (prefer sparse)
    B = np.asarray(pd_B_sub.to_numpy().T, dtype=dtype)  # genes x r

    X = adata_sub.X.T  # genes x N
    if sp.issparse(X):
        X = X.tocsr().astype(dtype)
    else:
        X = sp.csr_matrix(np.asarray(X, dtype=dtype))

    # Precompute GH + ||X||^2 for sparse-safe objective evaluation
    # G = X^T B: (N x r)
    G = X.T @ B
    if sp.issparse(G):
        G = G.toarray()
    G = np.asarray(G, dtype=dtype)

    # H = B^T B: (r x r)
    H = (B.T @ B).astype(dtype, copy=False)

    # ||X||_F^2
    xnorm2 = float(X.multiply(X).sum())

    def objective_from_GH(V, L):
        # ||X - B V^T||^2 = ||X||^2 - 2 tr(V^T G) + tr(V^T V H)
        VH = V @ H
        data = xnorm2 - 2.0 * float(np.sum(V * G)) + float(np.sum(VH * V))
        LV = L @ V
        reg = float(np.sum(LV * V))
        return data + reg

    # Initialize V per-slice if not provided
    if V_init is None:
        V_init = [None] * n_times
        for i in range(n_times):
            V_i = solve_V_per_slice(
                adata_group[i], pd_B_sub, lam=lam, k=k, eta=eta, step_rate=step_rate
            )
            V_i = np.asarray(V_i, dtype=dtype)
            if init == "random":
                V_i = rng.random(V_i.shape, dtype=float).astype(dtype)
                V_i = V_i / (V_i.sum(axis=1, keepdims=True) + 1e-15)
            V_init[i] = V_i
    else:
        # enforce dtype + list form
        V_init = [np.asarray(Vi, dtype=dtype) for Vi in V_init]

    V = np.vstack(V_init).astype(dtype, copy=False)  # (N x r)

    # Outer loop control
    if one_step:
        outer_max = 1
    if mu < 0:
        outer_max = -1

    # Spatial-only block Laplacian
    base_diag_blocks = [blk.copy() for blk in diag_blocks]
    L = assemble_blocks(base_diag_blocks, offdiag_blocks=[])
    if not sp.issparse(L):
        L = sp.csr_matrix(L)
    else:
        L = L.tocsr()

    f_prev = objective_from_GH(V, L)

    # store couplings between consecutive slices
    pi = [None] * (n_times - 1)

    for iter_outer in range(outer_max):
        # reset to spatial-only
        diag_blocks = [blk.copy() for blk in base_diag_blocks]
        off_diag_blocks = []

        # compute couplings, update blocks
        for i in range(n_times - 1):
            n_i = adata_group[i].n_obs
            n_ip = adata_group[i + 1].n_obs

            a = np.full(n_i, 1.0 / n_i, dtype=dtype)
            b = np.full(n_ip, 1.0 / n_ip, dtype=dtype)

            if coupling is None:
                # WARNING: cdist forms dense (n_i x n_ip) cost matrix
                C = scipy.spatial.distance.cdist(
                    V_init[i].astype(np.float64, copy=False),
                    V_init[i + 1].astype(np.float64, copy=False),
                    metric="sqeuclidean",
                )

                # spatial costs for FGW (also dense)
                x = adata_group[i].obs["x"].to_numpy()
                y = adata_group[i].obs["y"].to_numpy()
                C1 = ot.dist(np.vstack((x, y)).T)

                x = adata_group[i + 1].obs["x"].to_numpy()
                y = adata_group[i + 1].obs["y"].to_numpy()
                C2 = ot.dist(np.vstack((x, y)).T)

                # normalize
                C = C / (C.max() + 1e-15)
                C1 = C1 / (C1.max() + 1e-15)
                C2 = C2 / (C2.max() + 1e-15)

                if ot_solver == "fgw":
                    pi_i, log = fused_gromov_wasserstein(
                        C, C1, C2, a, b,
                        loss_fun="square_loss",
                        alpha=alpha,
                        log=True,
                    )
                    # fallback if coupling collapses
                    if np.std(pi_i.ravel()) < 1e-10:
                        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                        rr = min(C.shape[0], C.shape[1], 100)
                        if verbose:
                            print("constant coupling... fall back", C.shape)
                        pi_i, _ = FRLC.FRLC_opt(
                            torch.tensor(C, device=device),
                            A=torch.tensor(C1, device=device),
                            B=torch.tensor(C2, device=device),
                            alpha=alpha, device=device, r=rr,
                            min_iter=100, max_iter=100,
                            max_inneriters_balanced=300,
                            max_inneriters_relaxed=300,
                            min_iterGW=100,
                            Wasserstein=False, FGW=True,
                            returnFull=True, printCost=False,
                        )
                        pi_i = pi_i.detach().cpu().numpy()

                elif ot_solver == "sinkhorn":
                    geom_xy = PointCloud(
                        V_init[i]   / max(np.abs(Xj).max() for Xj in V_init),
                        V_init[i+1] / max(np.abs(Xj).max() for Xj in V_init),
                        epsilon=0.01,
                    )
                    problem = LinearProblem(geom_xy)
                    solver = Sinkhorn()
                    out_sink = solver(problem)
                    pi_i = np.array(out_sink.matrix)

                elif ot_solver == "sinkhorn_lr":
                    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                    rr = min(C.shape[0], C.shape[1], 100)
                    pi_i, _ = FRLC.FRLC_opt(
                        torch.tensor(C, device=device),
                        device=device, r=rr,
                        min_iter=100, max_iter=100,
                        max_inneriters_balanced=300,
                        max_inneriters_relaxed=300,
                        min_iterGW=100,
                        Wasserstein=True, FGW=False,
                        returnFull=True, printCost=False,
                    )
                    pi_i = pi_i.detach().cpu().numpy()

                else:
                    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                    rr = min(C.shape[0], C.shape[1], 100)
                    pi_i, _ = FRLC.FRLC_opt(
                        torch.tensor(C, device=device),
                        A=torch.tensor(C1, device=device),
                        B=torch.tensor(C2, device=device),
                        alpha=alpha, device=device, r=rr,
                        min_iter=100, max_iter=100,
                        max_inneriters_balanced=100,
                        max_inneriters_relaxed=100,
                        min_iterGW=100,
                        Wasserstein=False, FGW=True,
                        returnFull=True, printCost=False,
                    )
                    pi_i = pi_i.detach().cpu().numpy()

            else:
                pi_i = np.asarray(coupling[i], dtype=dtype)

            # normalize and time-scale (keep dense unless you sparsify pi explicitly)
            pi_i = pi_i / (pi_i.sum() + 1e-15)
            pi_i = pi_i * np.sqrt(pi_i.shape[0] * pi_i.shape[1])
            dt = (rescaled_time[i + 1] - rescaled_time[i])
            if dt != 0:
                pi_i = pi_i / dt

            pi[i] = pi_i

            a_marg = pi_i.sum(axis=1)  # (n_i,)
            b_marg = pi_i.sum(axis=0)  # (n_ip,)

            diag_blocks[i]   = diag_blocks[i]   + mu * sp.diags(a_marg, format="csr")
            diag_blocks[i+1] = diag_blocks[i+1] + mu * sp.diags(b_marg, format="csr")

            # off-diagonal dense blocks (WARNING: makes L less sparse if pi_i is dense)
            off_diag_blocks.append((i,   i+1, -mu * pi_i))
            off_diag_blocks.append((i+1, i,   -mu * pi_i.T))

        L = assemble_blocks(diag_blocks, offdiag_blocks=off_diag_blocks)
        if not sp.issparse(L):
            L = sp.csr_matrix(L)
        else:
            L = L.tocsr()

        if mu > 0:
            # IMPORTANT: solve_V_universal must be sparse-safe version
            V = solve_V_universal(
                X, B, L,
                niter_max=niter_max,
                tol=tol,
                eta=eta,
                verbose=verbose,
                V0=V,
                step_rate=step_rate,
            ).astype(dtype, copy=False)

            # update per-slice views
            for i in range(n_times):
                V_init[i] = V[n_spots[i]:n_spots[i+1], :]

        f_curr = objective_from_GH(V, L)
        rel_drop = (f_prev - f_curr) / max(1.0, abs(f_prev))

        if verbose and (iter_outer % 50 == 0 or rel_drop < tol):
            print(f"iter {iter_outer:4d}  f={f_curr:.6e}  rel_drop={rel_drop:.3e}")

        if f_curr <= f_prev and rel_drop < tol:
            if verbose:
                print("stopped after iteration #", iter_outer)
            break

        f_prev = f_curr

    return V_init, pi, L