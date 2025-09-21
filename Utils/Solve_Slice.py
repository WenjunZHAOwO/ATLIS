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
    sc.pp.log1p(adata)
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
from sklearn.metrics import pairwise_distances
from scipy.sparse import csgraph

def laplacian_from_coords(x, y, k=20, sigma=None, normed=True):
    """
    Construct a graph Laplacian from 2D coordinates.
    
    Parameters
    ----------
    x, y : array-like of shape (n,)
        Coordinates of points (spots).
    k : int
        Number of nearest neighbors to connect in the graph.
    sigma : float or None
        If given, use a Gaussian kernel exp(-d^2/(2*sigma^2)) for edge weights.
        If None, use unweighted adjacency (1 if neighbor).
    normed : bool
        If True, return normalized Laplacian (symmetric). Otherwise unnormalized.

    Returns
    -------
    L : (n, n) ndarray
        Graph Laplacian matrix.
    W : (n, n) ndarray
        Weight/adjacency matrix used.
    """
    coords = np.column_stack([x, y])
    D = pairwise_distances(coords)  # full distance matrix
    

    # Use median distance heuristic if sigma not provided
    if sigma is None:
        sigma = np.median(D[D > 0])/5
    n = len(x)
    W = np.zeros((n, n))
    for i in range(n):
        nn_idx = np.argsort(D[i])[1:k+1]  # skip self at [0]
        for j in nn_idx:
            d = D[i, j]
            w = np.exp(-d**2/(2*sigma**2))
            W[i, j] = W[j, i] = w

    # Degree matrix
    deg = np.sum(W, axis=1)
    Dmat = np.diag(deg)

    if normed:
        # symmetric normalized Laplacian: L = I - D^{-1/2} W D^{-1/2}
        with np.errstate(divide="ignore"):
            Dinv_sqrt = np.diag(1.0/np.sqrt(deg))
            Dinv_sqrt[np.isinf(Dinv_sqrt)] = 0
        L = np.eye(n) - Dinv_sqrt @ W @ Dinv_sqrt
    else:
        # unnormalized Laplacian: L = D - W
        L = Dmat - W

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


def solve_V_per_slice(adata, pd_B, lam=10, n_top=2000, niter_max=1e3, tol=1e-5, eta = 1e-3, verbose=True, preprocess=True):
    # stop
    if preprocess:
        sc.pp.log1p(adata)
        sc.pp.highly_variable_genes(adata, n_top_genes=n_top)

        mask = adata.var['highly_variable']
        gene_names = adata.var_names[mask]
        print(gene_names)
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

    L, W = laplacian_from_coords(x, y, k=20, sigma=None, normed=True)
    L = lam*L
    B = pd_B_sub.to_numpy().T
    X = adata_sub.X.T
    return solve_V_universal(X,B,L,niter_max=niter_max, tol=tol, eta = eta, verbose=verbose)

def solve_V_universal(X,B,L, niter_max=1e3, tol=1e-5, eta = 1e-3, verbose=True):

    print(X.T.shape)
    print(B.shape)

    G = X.T @ B
    H = B.T @ B
    rng = np.random.default_rng(0)

    d, n = X.shape[0], X.shape[1]
    r = B.shape[1]
    V = rng.random((n, r))
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
    return V




import anndata as ad
import ot

def solve_V_all_slices(adata_group, pd_B, lam=10, mu=1, n_top=2000, time=None, outer_max=5, niter_max=1e3, tol=1e-5, eta = 1e-3, verbose=True):
    # stop
    n_times = len(adata_group)
    n_spots = [0]
    for i in range(len(adata_group)):
        n_spots += [n_spots[-1]+adata_group[i].shape[0]]
    
    adata = ad.concat(adata_group, axis=0, join="outer", label="batch", keys=time)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, n_top_genes=n_top)

    mask = adata.var['highly_variable']
    gene_names = adata.var_names[mask]
    print(gene_names)
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
        diag_blocks[i], _ = laplacian_from_coords(x, y, k=20, sigma=None, normed=True)
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

            pi[i] = ot.sinkhorn(a,b,C, reg=1e-2)

            norm1 = np.linalg.norm(V_init[i],axis=1)
            norm2 = np.linalg.norm(V_init[i+1],axis=1)
            denom = np.outer(norm1, norm2)
            off_diag_blocks += [ (i,i+1,-mu*pi[i]/denom)  ]
            off_diag_blocks += [ (i+1,i,-mu*(pi[i]/denom).T)]

        L  = assemble_blocks(diag_blocks, offdiag_blocks=off_diag_blocks)
        V = solve_V_universal(X,B,L,niter_max=niter_max, tol=tol, eta = eta, verbose=verbose)
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


    

    return V_init, pi