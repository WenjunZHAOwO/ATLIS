import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, WhiteKernel, ConstantKernel
import sys
import FRLC
import torch

def gp_impute_spots(
    train_coords,          # (n_train, 2) array of [x, y]
    Vtrain,                # (n_train, n_ct) array of responses (e.g., proportions)
    test_coords,           # (n_test, 2) array of [x, y] where to impute
    kernel_type="RBF",     # "RBF" or "Matern"
    length_scale=50.0,     # initial spatial length scale (same units as coords)
    nu=1.5,                # Matern smoothness (only used if kernel_type="Matern")
    noise_level=1e-3,      # initial nugget (observation noise)
    optimize_hyperparams=True,   # if False, fix length_scale/noise at given values
    clip_to_01=True,       # clip predictions to [0,1]
    renormalize_rows=False, # if True, renormalize each row of Vtest to sum to 1
    return_std=False       # if True, also return predictive std per CT
):
    """
    Impute responses at new spatial locations with a GP per column of Vtrain.

    Returns
    -------
    Vtest : (n_test, n_ct) predictions
    Vstd  : (n_test, n_ct) predictive std (if return_std=True)
    """
    train_coords = np.asarray(train_coords, dtype=float)
    test_coords  = np.asarray(test_coords, dtype=float)
    Vtrain       = np.asarray(Vtrain, dtype=float)

    n_train, n_ct = Vtrain.shape
    n_test = test_coords.shape[0]

    if kernel_type.upper() == "RBF":
        base_k = RBF(length_scale=length_scale,
                     length_scale_bounds=(1e-3, 1e6) if optimize_hyperparams else "fixed")
    elif kernel_type.upper() == "MATERN":
        from sklearn.gaussian_process.kernels import Matern as SKMatern
        base_k = SKMatern(length_scale=length_scale, nu=nu,
                          length_scale_bounds=(1e-3, 1e6) if optimize_hyperparams else "fixed")
    else:
        raise ValueError("kernel_type must be 'RBF' or 'Matern'.")

    k_const = ConstantKernel(1.0,
                             constant_value_bounds=(1e-3, 1e3) if optimize_hyperparams else "fixed")
    k_noise = WhiteKernel(noise_level=noise_level,
                          noise_level_bounds=(1e-8, 1e2) if optimize_hyperparams else "fixed")

    kernel = k_const * base_k + k_noise

    Vtest = np.zeros((n_test, n_ct), dtype=float)
    Vstd  = np.zeros((n_test, n_ct), dtype=float) if return_std else None

    for c in range(n_ct):
        y = Vtrain[:, c]
        gpr = GaussianProcessRegressor(
            kernel=kernel,
            alpha=0.0,          # we already have WhiteKernel
            normalize_y=True,   # center/scale targets per CT
            n_restarts_optimizer=3 if optimize_hyperparams else 0,
            random_state=0
        )
        gpr.fit(train_coords, y)
        if return_std:
            pred, std = gpr.predict(test_coords, return_std=True)
            Vtest[:, c] = pred
            Vstd[:, c]  = std
        else:
            Vtest[:, c] = gpr.predict(test_coords, return_std=False)

    if clip_to_01:
        Vtest = np.clip(Vtest, 0.0, 1.0)

    if renormalize_rows:
        row_sums = Vtest.sum(axis=1, keepdims=True)
        # avoid divide-by-zero
        row_sums[row_sums == 0] = 1.0
        Vtest = Vtest / row_sums

    return (Vtest, Vstd) if return_std else Vtest



import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Kernel, RBF, WhiteKernel, ConstantKernel
from sklearn.neighbors import NearestNeighbors


import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern as SKMatern, WhiteKernel, ConstantKernel

def gp_impute_spots_cross_slice(
    train_coords, Vtrain, test_coords,
    kernel_type="RBF", length_scale=50.0, nu=1.5, noise_level=1e-3,
    optimize_hyperparams=True, clip_to_01=True, renormalize_rows=False, return_std=False,
    # NEW: optional cross-slice couplings (row-stochastic preferred)
    W_test_prev=None, V_prev=None,   # (n_test, N_prev), (N_prev, n_ct)
    W_test_next=None, V_next=None,   # (n_test, N_next), (N_next, n_ct)
    w_within=1.0, w_prev=0.0, w_next=0.0  # NEW: blend weights
):
    """
    Impute at test_coords with a GP per column of Vtrain, optionally blended with
    cross-slice transfers W_test_prev@V_prev and W_test_next@V_next.
    Works even if train_coords/Vtrain are empty (then uses couplings only).
    """
    test_coords  = np.asarray(test_coords, dtype=float)
    n_test = test_coords.shape[0]

    # --- Decide n_ct from any provided source
    n_ct = None
    if Vtrain is not None and len(Vtrain):
        Vtrain = np.asarray(Vtrain, dtype=float)
        n_ct = Vtrain.shape[1]
    if n_ct is None and V_prev is not None:
        n_ct = np.asarray(V_prev).shape[1]
    if n_ct is None and V_next is not None:
        n_ct = np.asarray(V_next).shape[1]
    if n_ct is None:
        raise ValueError("Cannot infer n_ct (need Vtrain or V_prev or V_next).")

    V_gp  = np.full((n_test, n_ct), np.nan)
    V_std = np.full((n_test, n_ct), np.nan) if return_std else None
    parts, weights = [], []

    # --- GP part (only if we have training points)
    has_train = (train_coords is not None) and (Vtrain is not None) and (len(train_coords) > 0)
    if has_train and w_within > 0:
        train_coords = np.asarray(train_coords, dtype=float)

        # kernel
        if kernel_type.upper() == "RBF":
            base_k = RBF(length_scale=length_scale,
                         length_scale_bounds=(1e-3, 1e6) if optimize_hyperparams else "fixed")
        elif kernel_type.upper() == "MATERN":
            base_k = SKMatern(length_scale=length_scale, nu=nu,
                              length_scale_bounds=(1e-3, 1e6) if optimize_hyperparams else "fixed")
        else:
            raise ValueError("kernel_type must be 'RBF' or 'Matern'.")

        k_const = ConstantKernel(1.0,
                                 constant_value_bounds=(1e-3, 1e3) if optimize_hyperparams else "fixed")
        k_noise = WhiteKernel(noise_level=noise_level,
                              noise_level_bounds=(1e-8, 1e2) if optimize_hyperparams else "fixed")
        kernel = k_const * base_k + k_noise

        for c in range(n_ct):
            y = Vtrain[:, c]
            gpr = GaussianProcessRegressor(
                kernel=kernel, alpha=0.0, normalize_y=True,
                n_restarts_optimizer=3 if optimize_hyperparams else 0, random_state=0
            )
            gpr.fit(train_coords, y)
            if return_std:
                mu, sd = gpr.predict(test_coords, return_std=True)
                V_gp[:, c] = mu; V_std[:, c] = sd
            else:
                V_gp[:, c] = gpr.predict(test_coords)
        parts.append(V_gp); weights.append(w_within)

    # --- Coupling parts (if provided)
    if (W_test_prev is not None) and (V_prev is not None) and (w_prev > 0):
        parts.append(np.asarray(W_test_prev, float) @ np.asarray(V_prev, float))
        weights.append(w_prev)
    if (W_test_next is not None) and (V_next is not None) and (w_next > 0):
        parts.append(np.asarray(W_test_next, float) @ np.asarray(V_next, float))
        weights.append(w_next)

    if not parts:
        raise ValueError("No sources available: provide training data and/or couplings.")

    # --- Blend
    Wsum = float(np.sum(weights))
    Vtest = sum(w * P for w, P in zip(weights, parts)) / Wsum

    if clip_to_01:
        Vtest = np.clip(Vtest, 0.0, 1.0)
    if renormalize_rows:
        s = Vtest.sum(axis=1, keepdims=True); s[s == 0] = 1.0
        Vtest = Vtest / s

    return (Vtest, V_std) if return_std else Vtest





import numpy as np
import ot
from ot.utils import dist
import matplotlib.pyplot as plt
import numpy as np
import ot
from ot.utils import dist
import matplotlib.pyplot as plt


def classical_mds(D, n_components=2):
    """
    Classical MDS from a distance matrix D.

    Parameters
    ----------
    D : (N, N) array
        Distance matrix (not squared).
    n_components : int
        Embedding dimension.

    Returns
    -------
    X : (N, n_components) array
        Embedded coordinates.
    """
    D = np.asarray(D)
    N = D.shape[0]
    # Double-centering
    J = np.eye(N) - np.ones((N, N)) / N
    B = -0.5 * J @ (D ** 2) @ J
    eigvals, eigvecs = np.linalg.eigh(B)
    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]
    eigvals_pos = np.clip(eigvals[:n_components], a_min=0, a_max=None)
    X = eigvecs[:, :n_components] * np.sqrt(eigvals_pos[np.newaxis, :])
    return X


def impute_slice_fgw_barycenter(
    X0,
    Y0,
    X1,
    Y1,
    t=0.5,
    N_bary=None,
    p0=None,
    p1=None,
    alpha=0.5,
    n_components=2,
    lambdas=None,
    loss_fun="square_loss",
    symmetric=True,
    max_iter=100,
    tol=1e-9,
    stop_criterion="barycenter",
    warmstartT=False,
    armijo=False,
    verbose=False,
    random_state=None,
    pd_B=None,
    Y_is_proportion=False,
    **kwargs,
):
    """
    fGW barycenter imputation with optional projection to gene space.

    If Y_is_proportion=True:
        - Y0, Y1, and the barycenter features are treated as proportions
        - barycenter features are projected via pd_B (cell type -> genes)

    If Y_is_proportion=False:
        - Y0, Y1 are already gene expression
        - output genes are reordered to pd_B.columns if provided
    """
    import numpy as np
    import ot
    from scipy.spatial.distance import cdist as dist

    X0 = np.asarray(X0)
    X1 = np.asarray(X1)
    Y0 = np.asarray(Y0)
    Y1 = np.asarray(Y1)

    assert X0.shape[0] == Y0.shape[0]
    assert X1.shape[0] == Y1.shape[0]

    n0, n1 = X0.shape[0], X1.shape[0]
    if N_bary is None:
        N_bary = int(np.round(0.5 * (n0 + n1)))

    # --- structural costs
    C0 = dist(X0, X0, metric="sqeuclidean")
    C1 = dist(X1, X1, metric="sqeuclidean")
    Cs = [C0, C1]
    Ys = [Y0, Y1]

    # --- measures
    if p0 is None:
        p0 = ot.utils.unif(n0)
    if p1 is None:
        p1 = ot.utils.unif(n1)
    ps = [p0, p1]

    # --- temporal weights
    if lambdas is None:
        lambdas = [1.0 - t, t]

    # --- fGW barycenter
    X_bar_feat, C_bar, log = ot.gromov.fgw_barycenters(
        N=N_bary,
        Ys=Ys,
        Cs=Cs,
        ps=ps,
        lambdas=lambdas,
        alpha=alpha,
        fixed_structure=False,
        fixed_features=False,
        p=None,
        loss_fun=loss_fun,
        armijo=armijo,
        symmetric=symmetric,
        max_iter=max_iter,
        tol=tol,
        stop_criterion=stop_criterion,
        warmstartT=warmstartT,
        verbose=verbose,
        log=True,
        init_C=None,
        init_X=None,
        random_state=random_state,
        **kwargs,
    )

    # --- embed barycenter geometry
    D_bar = np.sqrt(np.maximum(C_bar, 0.0))
    X_bar_spatial = classical_mds(D_bar, n_components=n_components)

    # =========================================================
    # NEW: project to gene expression if requested
    # =========================================================
    if Y_is_proportion:
        if pd_B is None:
            raise ValueError("pd_B must be provided when Y_is_proportion=True.")

        # pd_B: (n_cell_types, n_genes)
        X_bar_gene = X_bar_feat @ pd_B.values

        return (
            X_bar_spatial,
            pd.DataFrame(X_bar_gene, columns=pd_B.columns),
            X_bar_feat,
            C_bar,
            log,
        )

    # already gene expression → reorder columns if pd_B given
    if pd_B is not None:
        if X_bar_feat.shape[1] != pd_B.shape[1]:
            raise ValueError("Gene dimension mismatch with pd_B.")
        X_bar_feat = pd.DataFrame(X_bar_feat, columns=pd_B.columns)

    return X_bar_spatial, X_bar_feat, C_bar, log



import numpy as np
import ot
from scipy.spatial.distance import cdist


import numpy as np
import pandas as pd
import ot
from scipy.spatial.distance import cdist


def impute_slice_gw_with_projection(
    coords_prev, X_prev,
    coords_t,
    coords_next, X_next,
    alpha=0.5,
    eps=5e-3,
    pd_B=None,
    X_is_proportion=False,
    solver='fgw',
    coords_is_dist= False,
):
    """
    GW-based full-slice imputation.
    Optionally projects proportions to gene expression using pd_B.

    Parameters
    ----------
    coords_prev, coords_t, coords_next : (N, d)
    X_prev, X_next : (N_prev, p) and (N_next, p)
        Either gene expression or cell-type proportions.
    alpha : float
        Time interpolation weight.
    eps : float
        Entropic GW regularization.
    pd_B : pandas.DataFrame or None
        Rows = cell types, columns = genes.
    X_is_proportion : bool
        True if X_* are proportions over pd_B.index.

    Returns
    -------
    X_t_hat : (N_t, n_genes) or (N_t, p)
    """

    # --- GW coupling
    def _gw(coords_t, coords_s, coords_is_dist):
        
        Ct = cdist(coords_t, coords_t, metric="sqeuclidean")
        if coords_is_dist == False:
            Cs = cdist(coords_s, coords_s, metric="sqeuclidean")
            C = cdist(coords_s, coords_t, metric="sqeuclidean")
        else:
            Cs = coords_s
            C = np.zeros((Cs.shape[0], Ct.shape[0]))
        Cs = Cs/Ct.max()
        Ct = Ct/Ct.max()
        
        pt = np.ones(Ct.shape[0]) / Ct.shape[0]
        ps = np.ones(Cs.shape[0]) / Cs.shape[0]
        # C = cdist(coords_s, coords_t, metric="sqeuclidean")
        if solver == 'fgw':
            G = ot.gromov.entropic_gromov_wasserstein(
                Ct, Cs, pt, ps, epsilon=eps, loss_fun="square_loss"
            )
            if np.std(G.ravel()) < 1e-10:
                print('degenerate coupling...switch to lowrank')
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                G, _ = FRLC.FRLC_opt(torch.tensor(C).to(device), A = torch.tensor(Cs).to(device), B = torch.tensor(Ct).to(device), alpha=1.0, device=device, r=100, min_iter=100, max_iter=100,
                                    max_inneriters_balanced=300,
                                    max_inneriters_relaxed=300,
                                    min_iterGW = 100,
                                    Wasserstein = False,
                                    FGW = True,
                                    returnFull=True,
                                    printCost=False
                )
                G = G.cpu().detach().numpy()
                G = G.T
        else:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            G, _ = FRLC.FRLC_opt(torch.tensor(C).to(device), A = torch.tensor(Cs).to(device), B = torch.tensor(Ct).to(device), alpha=1.0, device=device, r=100, min_iter=100, max_iter=100,
                                    max_inneriters_balanced=300,
                                    max_inneriters_relaxed=300,
                                    min_iterGW = 100,
                                    Wasserstein = False,
                                    FGW = True,
                                    returnFull=True,
                                    printCost=False
            )
            G = G.cpu().detach().numpy()
            G = G.T
        return G / G.sum(axis=1, keepdims=True)

    
    W_prev = _gw(coords_t, coords_prev, coords_is_dist)
    X_prev = np.asarray(X_prev, float)
    X_t_hat = W_prev @ X_prev
    if not coords_next is None:

        W_next = _gw(coords_t, coords_next, coords_is_dist)
        X_next = np.asarray(X_next, float)

        X_t_hat = (1 - alpha) * (W_prev @ X_prev) + alpha * (W_next @ X_next)

    # --- If proportions → genes
    if X_is_proportion:
        if pd_B is None:
            raise ValueError("pd_B must be provided when X_is_proportion=True.")

        # pd_B: (n_ct, n_genes)
        B = pd_B.values
        X_t_prop = X_t_hat.copy()
        X_t_hat = X_t_hat @ B   # (N_t, n_genes)

        return pd.DataFrame(
            X_t_hat,
            columns=pd_B.columns
        ), X_t_prop

    # --- If already genes, just reorder to pd_B.columns if provided
    if pd_B is not None:
        if X_t_hat.shape[1] != pd_B.shape[1]:
            raise ValueError("Gene dimension mismatch with pd_B columns.")
        return pd.DataFrame(
            X_t_hat,
            columns=pd_B.columns
        )

    return X_t_hat


import numpy as np
import ot

from sklearn.metrics import pairwise_distances

def fps_sklearn(X, n_max=1000, random_state=0):
    n = X.shape[0]
    rng = np.random.RandomState(random_state)

    idx = np.empty(n_max, dtype=int)
    idx[0] = rng.randint(n)

    dist = pairwise_distances(X, X[idx[0]][None, :]).ravel()

    for k in range(1, n_max):
        idx[k] = np.argmax(dist)
        dist = np.minimum(
            dist,
            pairwise_distances(X, X[idx[k]][None, :]).ravel()
        )

    return idx


def _downsample_pair(X, Y=None, n_max=None, rng=None):
    """
    Downsample rows of X and Y together to at most n_max points.
    Keeps alignment. Returns (X_ds, Y_ds).
    """
    if n_max is None or X.shape[0] <= n_max:
        return X, Y
    if rng is None:
        rng = np.random.default_rng(0)
    idx = fps_sklearn(X, n_max=n_max)
    # idx = rng.choice(X.shape[0], size=int(n_max), replace=False)


    if Y is None:
        return X[idx]
    else:
        return X[idx], Y[idx]

def compare_imputation_midpoint(adata_insamp, V, pd_B, i, n_end=1000, type_id=-1, depth=4):
    """
    depth controls how fine the recursive refinement is:
      depth=1 -> t in {0.5}
      depth=2 -> t in {0.25, 0.5, 0.75}
      depth=3 -> t in {0.125, 0.25, ..., 0.875}
    """

    def coords(ad):
        return ad.obs[["x", "y"]].to_numpy(dtype=float)

    eps = 1e0
    ad0, adt, ad1 = adata_insamp[i - 1], adata_insamp[i], adata_insamp[i + 1]

    V0 = np.asarray(V[i - 1], float)
    Vt = np.asarray(V[i], float)
    V1 = np.asarray(V[i + 1], float)

    X0 = coords(ad0)
    Xt = coords(adt)
    X1 = coords(ad1)

    # keep your original containers
    X_all = [[X0], [Xt], [X1]]
    V_all = [[V0], [Vt], [V1]]



    # Downsample endpoints once for speed
    X0_prop_ds, Y0_prop_ds = _downsample_pair(X0, V0, n_max=n_end)
    X1_prop_ds, Y1_prop_ds = _downsample_pair(X1, V1, n_max=n_end)

    # ---------- Option (2): recursive midpoint refinement ----------
    # Store nodes keyed by exact dyadic t as integers to avoid float-key issues:
    # represent t = k / 2^depth by integer k in [0, 2^depth]
    L = int(depth)
    denom = 2**L

    # nodes[k] = (Xk, Yk, Ck) where t = k/denom
    nodes = {}
    nodes[0] = (X0_prop_ds, Y0_prop_ds, ot.dist(X0_prop_ds, X0_prop_ds))
    nodes[denom] = (X1_prop_ds, Y1_prop_ds, ot.dist(X1_prop_ds, X1_prop_ds))

    def _midpoint_node(left_k, right_k):
        """Compute node at mid_k = (left_k+right_k)//2 using parents at t=0.5."""
        mid_k = (left_k + right_k) // 2
        if mid_k in nodes:
            return mid_k

        Xl, Yl, _ = nodes[left_k]
        Xr, Yr, _ = nodes[right_k]

        Xbar, _, Ybar, Cbar, _ = impute_slice_fgw_barycenter(
            X0=Xl, Y0=Yl,
            X1=Xr, Y1=Yr,
            t=0.5,                 # midpoint interpolation
            N_bary=n_end,
            alpha=0.5,
            pd_B=pd_B,
            Y_is_proportion=True,
        )
        nodes[mid_k] = (Xbar, Ybar, Cbar)
        return mid_k

    # Build dyadic points by recursively splitting intervals
    intervals = [(0, denom)]
    for _ in range(L):  # each level splits all current intervals
        new_intervals = []
        for a, b in intervals:
            print('a='+str(a)+',b='+str(b))
            if b - a <= 1:
                continue
            m = _midpoint_node(a, b)
            new_intervals.append((a, m))
            new_intervals.append((m, b))
        intervals = new_intervals

    # Now evaluate GW distance for all interior dyadic points
    # (you can include endpoints too if you want, but typically exclude 0 and 1)
    ks = sorted([k for k in nodes.keys() if 0 < k < denom])
    t_grid = [0] + [k / denom for k in ks] + [1]

    gw_dists = []
    best_gw = np.inf
    best_t = None
    best_Xbar = None
    best_Ybar = None
    best_Cbar = None

    # target geometry cost (fixed)

    q = np.ones(Xt.shape[0]) / Xt.shape[0]
    X_all += [X0_prop_ds]
    V_all += [Y0_prop_ds]

    C2 = ot.dist(Xt, Xt)
    C1 = ot.dist(X0_prop_ds, X0_prop_ds)
    C1 = C1 / (C2.max() + 1e-12)  # consistent scaling with your previous code pattern
    C2 = C2 / (C2.max() + 1e-12)

    p = np.ones(C1.shape[0]) / C1.shape[0]
    gw = ot.gromov.gromov_wasserstein2(C1, C2, p, q, loss_fun="square_loss")
    gw_dists.append(gw)

    for k in ks:
        t = k / denom
        print(f"t = {t:.6f}")

        Xbar, Ybar, Cbar = nodes[k]
        
        # keep for plotting (same as your earlier usage: append arrays, not nested)
        X_all += [Xbar]
        V_all += [Ybar]

        # GW distance to evaluation slice
        C2 = ot.dist(Xt, Xt)
        C1 = Cbar
        C1 = C1 / (C2.max() + 1e-12)  # consistent scaling with your previous code pattern
        C2 = C2 / (C2.max() + 1e-12)

        p = np.ones(Xbar.shape[0]) / Xbar.shape[0]
        gw = ot.gromov.gromov_wasserstein2(C1, C2, p, q, loss_fun="square_loss")

        gw_dists.append(gw)
        print("GW =", gw)

        if gw < best_gw:
            best_gw = gw
            best_t = float(t)
            best_Xbar = Xbar
            best_Ybar = Ybar
            best_Cbar = Cbar
    X_all += [X1_prop_ds]
    V_all += [Y1_prop_ds]
    C2 = ot.dist(Xt, Xt)
    C1 = ot.dist(X1_prop_ds, X1_prop_ds)
    C1 = C1 / (C2.max() + 1e-12)  # consistent scaling with your previous code pattern
    C2 = C2 / (C2.max() + 1e-12)

    p = np.ones(C1.shape[0]) / C1.shape[0]
    gw = ot.gromov.gromov_wasserstein2(C1, C2, p, q, loss_fun="square_loss")
    gw_dists.append(gw)



    return X_all, V_all, t_grid, gw_dists


import numpy as np

def _rot2(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s],
                     [s,  c]])

def _pca_angle(X):
    Xc = X - X.mean(axis=0, keepdims=True)
    S = (Xc.T @ Xc) / max(len(Xc) - 1, 1)
    vals, vecs = np.linalg.eigh(S)
    v = vecs[:, np.argmax(vals)]
    return float(np.arctan2(v[1], v[0]))

def _chamfer(A, B):
    A = np.asarray(A, float)
    B = np.asarray(B, float)
    d = ((A[:, None, :] - B[None, :, :])**2).sum(axis=2)
    return float(d.min(axis=1).mean() + d.min(axis=0).mean())

def pca_align_then_temporal_flip(X, X_prev_aligned=None, fix_scale=False, return_info=False):
    """
    PCA-align X (center + rotate) then resolve the 4-way ambiguity by choosing
    the candidate that is closest to the previous aligned slice X_prev_aligned.
    If X_prev_aligned is None, returns PCA-aligned (no flip decision).
    """
    X = np.asarray(X, float)

    # center + optional scale
    c = X.mean(axis=0, keepdims=True)
    Xc = X - c
    s = 1.0
    if fix_scale:
        s = np.sqrt((Xc**2).sum(axis=1).mean()) + 1e-12
        Xc = Xc / s

    # PCA rotate so PC1 is horizontal
    theta = _pca_angle(Xc)
    Xp = Xc @ _rot2(-theta).T

    # if no previous, do nothing further
    if X_prev_aligned is None:
        X_out = (Xp * s + c) if fix_scale else (Xp + c)
        if not return_info:
            return X_out
        return X_out, {"theta_rad": theta, "chosen_flip": (1.0, 1.0), "used_prev": False}

    # build 4 candidates: identity, flip-x, flip-y, flip-both (== 180°)
    flips = [(1.0, 1.0), (-1.0, 1.0), (1.0, -1.0), (-1.0, -1.0)]
    cands = [Xp * np.array([fx, fy]) for fx, fy in flips]

    # compare in the same normalization as output
    # (centered; scale-normalized if fix_scale)
    prev = np.asarray(X_prev_aligned, float)
    prevc = prev - prev.mean(axis=0, keepdims=True)
    if fix_scale:
        prevs = np.sqrt((prevc**2).sum(axis=1).mean()) + 1e-12
        prevc = prevc / prevs

    dists = [_chamfer(C, prevc) for C in cands]
    best = int(np.argmin(dists))
    Xbest = cands[best]
    fx, fy = flips[best]

    # unscale/uncenter back to original frame (or keep centered if you prefer)
    X_out = Xbest
    if fix_scale:
        X_out = X_out * s
    X_out = X_out + c

    if not return_info:
        return X_out

    return X_out, {
        "theta_rad": float(theta),
        "chosen_flip": (float(fx), float(fy)),
        "used_prev": True,
        "dists_to_prev": list(map(float, dists)),
    }


def rotate_points(XY, angle_deg, center=None):
    theta = np.deg2rad(angle_deg)
    R = np.array([[np.cos(theta), -np.sin(theta)],
                  [np.sin(theta),  np.cos(theta)]])

    if center is not None:
        XY = XY - np.asarray(center)

    XY_rot = XY @ R.T

    if center is not None:
        XY_rot = XY_rot + np.asarray(center)

    return XY_rot