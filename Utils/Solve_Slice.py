import numpy as np
import scipy
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Wedge, Patch
import scanpy as sc
from scipy.sparse import lil_matrix, csr_matrix

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
    X = adata.X
    if not isinstance(X, np.ndarray):
        X = X.toarray()   # optional; you can also work with sparse

    # Create a DataFrame: rows=cells, cols=genes
    expr_df = pd.DataFrame(X, index=adata.obs.index, columns=adata.var_names)

    # Add cell type column
    expr_df['cellType'] = adata.obs['cellType'].values

    # Group by cell type and compute mean expression across cells
    avg_expr = expr_df.groupby('cellType').mean()

    return avg_expr

import numpy as np
import pandas as pd
from typing import Optional, Tuple, Dict, Any
from scipy import sparse

import numpy as np
import pandas as pd

def select_informative_genes(B,
                             fc_thresh_log2=1.25) -> list:
    
    eps = 1e-6
    fc_thresh_ln = fc_thresh_log2 * np.log(2.0)  # convert log2 FC to ln

    gene_set = set()
    for ict in B.index:
        rest = B.drop(index=ict).mean(axis=0)
        FC = np.log(B.loc[ict] + eps) - np.log(rest + eps)
        keep = (FC > fc_thresh_ln) & (B.loc[ict] > 0)
        gene_set.update(B.columns[keep])

    return sorted(gene_set)


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

def laplacian_from_coords(x, y, k=20, mutual=False, normed=True, return_dense=False):
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

import numpy as np

def proj_simplex_rows(V, z=1.0):
    """
    Project each row of M onto the simplex { x : x >= 0, sum(x) = z }.

    Parameters
    ----------
    V : array_like, shape (n_rows, n_cols)
        Input matrix; each row will be projected.
    z : float, optional
        Sum constraint for each row (default 1.0).

    Returns
    -------
    X : ndarray, shape (n_rows, n_cols)
        Row-wise simplex projections.
    """
    V = np.asarray(V, dtype=float)
    n, m = V.shape

    # sort each row in descending order
    U = -np.sort(-V, axis=1)                        # (n, m)
    cssv = np.cumsum(U, axis=1)                     # row-wise cumsum

    # find rho per row: largest j s.t. u_j * (j+1) > cssv_j - z
    j = np.arange(1, m+1)
    cond = (U * j) > (cssv - z)                     # (n, m) boolean
    rho = cond.sum(axis=1) - 1                      # (n,) index

    # theta per row
    theta = (cssv[np.arange(n), rho] - z) / (rho + 1.0)  # (n,)

    # projection
    X = V - theta[:, None]
    np.maximum(X, 0.0, out=X)
    return X


def scatter_pies(ax, x, y, percents, radius=0.35, labels=None,
                 colors=None, start_angle=90, edgecolor="white", lw=0.2,
                 top_n=None):
    """
    Draw a pie chart at each (x,y).

    Parameters
    ----------
    x, y : array-like
        Coordinates.
    percents : (n_spots, k) array
        Fractions for each cell type per spot (will be row-normalized).
    labels : list of str, optional
        Cell type names for columns.
    top_n : int, optional
        If given, only keep top_n cell types by average abundance.
    """
    x = np.asarray(x); y = np.asarray(y)
    P = np.asarray(percents, float)
    P = np.nan_to_num(P, nan=0.0)

    # Row-normalize
    rs = P.sum(axis=1, keepdims=True); rs[rs == 0] = 1.0
    P = P / rs

    n, k = P.shape
    if labels is None:
        labels = [f"C{i+1}" for i in range(k)]

    # --- keep only top_n cell types ---
    if top_n is not None and top_n < k:
        avg = P.mean(axis=0)
        keep_idx = np.argsort(avg)[::-1][:top_n]
        P = P[:, keep_idx]
        labels = [labels[i] for i in keep_idx]
        k = top_n

    # Colors
    if colors is None:
        cmap = plt.get_cmap("tab20")
        colors = [cmap(i % 20) for i in range(k)]
    else:
        colors = [colors[i] for i in keep_idx] if (top_n is not None) else colors

    # --- draw pies ---
    for xi, yi, fracs in zip(x, y, P):
        theta = start_angle
        for f, c in zip(fracs, colors):
            if f <= 0:
                continue
            w = Wedge((xi, yi), r=radius,
                      theta1=theta, theta2=theta + 360.0*f,
                      facecolor=c, edgecolor=edgecolor, linewidth=lw)
            ax.add_patch(w)
            theta += 360.0*f

    # --- adjust view ---
    ax.set_xlim(x.min()-radius*1.1, x.max()+radius*1.1)
    ax.set_ylim(y.min()-radius*1.1, y.max()+radius*1.1)
    ax.set_aspect("equal", adjustable="box")

    # --- legend ---
    handles = [Patch(facecolor=colors[i], edgecolor=edgecolor, label=labels[i]) for i in range(k)]
    ax.legend(
        handles=handles,
        title="Cell types",
        frameon=False,
        loc="center left",
        bbox_to_anchor=(1.02, 0.5)
    )
    return handles


def solve_V_per_slice(adata, pd_B, lam=10, n_top=2000, niter_max=1e3, tol=1e-5, eta = 1e-3, verbose=True, preprocess=True, return_obj=False):
    # stop
    if preprocess:
        import scipy.sparse as sp
        # --- 1) Remove non-finite values in X (works for dense or sparse) ---
        if sp.issparse(adata.X):
            d = adata.X.data
            bad = ~np.isfinite(d)
            if bad.any():
                d[bad] = 0.0
        else:
            adata.X = np.nan_to_num(adata.X, nan=0.0, posinf=0.0, neginf=0.0)

        # (Optional) also sanitize any layers to be safe
        for lyr in getattr(adata, "layers", {}) or {}:
            Xl = adata.layers[lyr]
            if sp.issparse(Xl):
                dl = Xl.data
                badl = ~np.isfinite(dl)
                if badl.any():
                    dl[badl] = 0.0
            else:
                adata.layers[lyr] = np.nan_to_num(Xl, nan=0.0, posinf=0.0, neginf=0.0)

        # --- 2) Basic preproc (HVG expects normalized/log data) ---
        sc.pp.filter_genes(adata, min_counts=1)
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.filter_genes(adata, min_counts=1)
        sc.pp.highly_variable_genes(adata, n_top_genes=n_top)
        mask = adata.var['highly_variable']
        gene_names = adata.var_names[mask]

        print(pd_B.columns)
        # Intersection with B’s index (genes as rows)
        common = gene_names.intersection(pd_B.columns)
        # print(common)
        # Subset AnnData and B, in the same order
        adata_sub = adata[:, common].copy()
        pd_B_sub = pd_B[common]
    
    else:
        adata_sub = adata.copy()
        pd_B_sub = pd_B
    x = adata_sub.obs['x'].to_numpy()
    y = adata_sub.obs['y'].to_numpy()

    L, W = laplacian_from_coords(x, y, k=20, normed=True)
    L = lam*L
    B = pd_B_sub.to_numpy().T
    X = adata_sub.X.T
    return solve_V_universal(X,B,L,niter_max=niter_max, tol=tol, eta = eta, verbose=verbose, return_obj=return_obj)

def solve_V_universal(X,B,L, niter_max=1e3, tol=1e-5, eta = 1e-3, verbose=True, V0=None, return_obj=False):

    print(X.T.shape)
    print(B.shape)

    G = X.T @ B
    H = B.T @ B
    rng = np.random.default_rng(0)

    d, n = X.shape[0], X.shape[1]
    r = B.shape[1]
    if V0 is None:
        V = rng.random((n, r))
    else:
        V = V0
    V = V / V.sum(axis=1, keepdims=True)

    

    def objective(V):
        R = X - B @ V.T
        R = np.asarray(R)
        return np.sum(R*R) + np.linalg.trace(V.T @ L @ V)
    
    def deriv_objective(V, H, G):
        return 2*(V @ H - G) + 2*L@V
    

    f_prev = objective(V)
    for it in range(int(niter_max)):
        g = deriv_objective(V, H, G)   # gradient
        V0 = V.copy()

        step = eta   # start with base step each iteration
        while True:
            # candidate update
            V_trial = proj_simplex_rows(V0 - step * g, z=1.0)
            f_trial = objective(V_trial)

            # Armijo condition (sufficient decrease)
            # here alpha is small (e.g. 1e-4)
            if f_trial <= f_prev - 1e-4 * step * np.sum(g*g):
                V = V_trial
                f_curr = f_trial
                break
            step *= 0.5
            if step < 1e-8:  # step too small, give up
                V = V_trial
                f_curr = f_trial
                break

        # stopping checks
        rel_drop = (f_prev - f_curr) / max(1.0, abs(f_prev))
        if verbose and (it % 50 == 0 or rel_drop < tol):
            print(f"iter {it:4d}  f={f_curr:.6e}  rel_drop={rel_drop:.3e}  step={step:.2e}")

        if f_curr <= f_prev and rel_drop < tol:
            print("stopped after iteration #", it)
            break

        f_prev = f_curr

    print('stopped after iteration #'+str(it))
    if return_obj == False:
        return V
    else:
        R = X - B @ V.T
        R = np.asarray(R)
        return V, objective(V), np.sum(R*R)
        




import anndata as ad
import ot

def solve_V_all_slices(adata_group, pd_B, lam=10, mu=1, n_top=2000, time=None, outer_max=5, niter_max=1e3, tol=1e-5, eta = 1e-3, verbose=True, preprocess=True, coupling=None):
    # stop
    n_times = len(adata_group)
    n_spots = [0]
    for i in range(len(adata_group)):
        n_spots += [n_spots[-1]+adata_group[i].shape[0]]
    
    adata = ad.concat(adata_group, axis=0, join="outer", label="batch", keys=time)
    if preprocess:
        import scipy.sparse as sp
        # --- 1) Remove non-finite values in X (works for dense or sparse) ---
        if sp.issparse(adata.X):
            d = adata.X.data
            bad = ~np.isfinite(d)
            if bad.any():
                d[bad] = 0.0
        else:
            adata.X = np.nan_to_num(adata.X, nan=0.0, posinf=0.0, neginf=0.0)

        # (Optional) also sanitize any layers to be safe
        for lyr in getattr(adata, "layers", {}) or {}:
            Xl = adata.layers[lyr]
            if sp.issparse(Xl):
                dl = Xl.data
                badl = ~np.isfinite(dl)
                if badl.any():
                    dl[badl] = 0.0
            else:
                adata.layers[lyr] = np.nan_to_num(Xl, nan=0.0, posinf=0.0, neginf=0.0)

        # --- 2) Basic preproc (HVG expects normalized/log data) ---
        sc.pp.filter_genes(adata, min_counts=1)
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.filter_genes(adata, min_counts=1)
        sc.pp.highly_variable_genes(adata, n_top_genes=n_top)
        mask = adata.var['highly_variable']
        gene_names = adata.var_names[mask]
    else:
        gene_names = adata.var_names


    print(pd_B.columns)
    # Intersection with B’s index (genes as rows)
    common = gene_names.intersection(pd_B.columns)
    # print(common)
    # Subset AnnData and B, in the same order
    adata_sub = adata[:, common].copy()
    for i in range(n_times):
        adata_group[i] = adata_group[i][:,common].copy()
    pd_B_sub = pd_B[common]
    

    diag_blocks = [0]*n_times
    for i in range(n_times):
        x = adata_group[i].obs['x'].to_numpy()
        y = adata_group[i].obs['y'].to_numpy()
        diag_blocks[i], _ = laplacian_from_coords(x, y, k=20, normed=True)
        diag_blocks[i] = diag_blocks[i]*lam

    B = pd_B_sub.to_numpy().T
    X = adata_sub.X.T
    V_init = [0]*n_times
    for i in range(n_times):
        V_init[i] = solve_V_per_slice(adata_group[i],pd_B_sub, lam=lam, preprocess=False)
    
    def objective(V,L):
        R = X - B @ V.T
        R = np.asarray(R)
        return np.sum(R*R) + np.linalg.trace(V.T @ L @ V)
    
    V = np.vstack(V_init)
    L  = assemble_blocks(diag_blocks, offdiag_blocks=[])#off_diag_blocks)
    
    f_prev = objective(V,L)
    pi = [0]*n_times

    for iter_outer in range(outer_max):
        off_diag_blocks = []
        for i in range(n_times-1):
            a = np.ones(adata_group[i].obs.shape[0])/adata_group[i].obs.shape[0]
            b = np.ones(adata_group[i+1].obs.shape[0])/adata_group[i+1].obs.shape[0]
            C = scipy.spatial.distance.cdist(V_init[i], V_init[i+1], metric='cosine')
            if coupling is None:
                pi[i] = ot.sinkhorn(a,b,C, reg=1e-2)
            else:
                pi[i] = coupling[i]

            norm1 = np.linalg.norm(V_init[i],axis=1)
            norm2 = np.linalg.norm(V_init[i+1],axis=1)

            # norm1[norm1 == 0] = 1e-10
            # norm2[norm2 == 0] = 1e-10
            denom = np.outer(norm1, norm2)
            denom[denom == 0] = 1e-10
            off_diag_blocks += [ (i,i+1,-mu*pi[i]/denom)  ]
            off_diag_blocks += [ (i+1,i,-mu*(pi[i]/denom).T)]
        L0 = L.copy()
        L  = assemble_blocks(diag_blocks, offdiag_blocks=off_diag_blocks)
        
        plt.subplot(121)
        plt.imshow(L.todense())
        plt.clim([0,1])
        plt.colorbar()
        plt.subplot(122)
        plt.imshow(L.todense() - L0.todense())
        plt.colorbar()
        plt.clim([0,1])
        # stop

        V = solve_V_universal(X,B,L,niter_max=niter_max, tol=tol, eta = eta, verbose=verbose)#, V0 = V)
        f_curr = objective(V,L)
        for i in range(n_times-1):
            V_init[i] = V[n_spots[i]:n_spots[i+1],:]
    # stopping checks
        rel_drop = (f_prev - f_curr) / max(1.0, abs(f_prev))
        if verbose and (iter_outer % 50 == 0 or rel_drop < tol):
            print(f"iter {iter_outer:4d}  f={f_curr:.6e}  rel_drop={rel_drop:.3e}")

        if f_curr <= f_prev and rel_drop < tol:
            print("stopped after iteration #", iter_outer)
            break

        f_prev = f_curr


    

    return V_init, pi, L