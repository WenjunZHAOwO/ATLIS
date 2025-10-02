import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, WhiteKernel, ConstantKernel

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
