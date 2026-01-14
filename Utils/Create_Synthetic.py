import sys

import Solve_Slice_test
from Solve_Slice_test import solve_V_per_slice, laplacian_from_coords, solve_V_all_slices, find_B_music
import anndata as ad
import numpy as np
import pandas as pd


def create_data(noise=2, n_time=1, 
                stripe=False, 
                stripe_width=3.0, 
                stripe_type=2,        # 0=Type1,1=Type2,2=Type3
                stripe_axis="x",
               seed = 0):     # "x" = vertical stripe, "y" = horizontal
    rng = np.random.default_rng(seed)
    adata = ad.read_h5ad("../Data/spatial.h5ad")   # replace with your filename
    spatial_coords = np.vstack([adata.obs['x'], adata.obs['y']])  # shape (2, n_spots)
    center = np.mean(spatial_coords, axis=1)

    # Cell-type-by-gene matrix B
    pd_B = pd.DataFrame(
        data=[[0, 1, 0],
              [1, 0, 0],
              [0.5, 0.0, 1.0]],
        columns=['Gene 1', 'Gene 2', 'Gene 3'],
        index=['Type 1', 'Type 2', 'Type 3']
    )

    # ---- static V_true block (kept same as your code) ----
    V_true = np.zeros((spatial_coords.shape[1], pd_B.shape[0]))

    V_true[:, 0] = np.exp(-(spatial_coords[0] - center[0]) ** 2 / 5)
    V_true[:, 1] = np.exp(-(spatial_coords[1] - center[1]) ** 2 / 5)
    V_true[:, 2] = 2 * np.exp(-(spatial_coords[0] - center[0]) ** 2 / 7) * \
                      np.exp(-(spatial_coords[1] - center[1]) ** 2 / 7)
    V_true = V_true / np.mean(V_true, axis=1).reshape(-1, 1)

    counts = V_true @ pd_B.values + np.random.rand(V_true.shape[0], pd_B.shape[1]) * noise
    counts[counts < 0] = 0
    adata_synthetic = ad.AnnData(counts)
    adata_synthetic.obs['x'] = adata.obs['x'].values
    adata_synthetic.obs['y'] = adata.obs['y'].values
    adata_synthetic.obs_names = [f"Cell_{i:d}" for i in range(adata_synthetic.n_obs)]
    adata_synthetic.var_names = [f"Gene {i+1:d}" for i in range(adata_synthetic.n_vars)]

    # ---- temporal part ----
    lower_left = [0, 0]
    t = range(n_time)

    # ---- compute stripe mask if requested ----
    if stripe:
        if stripe_axis == "x":
            stripe_center = np.median(spatial_coords[0])
            stripe_mask = np.abs(spatial_coords[0] - stripe_center) < stripe_width
        elif stripe_axis == "y":
            stripe_center = np.median(spatial_coords[1])
            stripe_mask = np.abs(spatial_coords[1] - stripe_center) < stripe_width
        else:
            raise ValueError("stripe_axis must be 'x' or 'y'.")
    else:
        stripe_mask = None

    V_true_temporal = [0] * len(t)
    for i in t:
        print(i)
        V_tmp = np.zeros_like(V_true)

        cx_t = i / n_time * center[0] + (1 - (i+1) / n_time) * lower_left[0]
        cy_t = i / n_time * center[1] + (1 - (i+1) / n_time) * lower_left[1]

        V_tmp[:, 0] = np.exp(-(spatial_coords[0] - cx_t) ** 2 / 5)
        V_tmp[:, 1] = np.exp(-(spatial_coords[1] - cy_t) ** 2 / 5)
        V_tmp[:, 2] = 2 * np.exp(-(spatial_coords[0] - center[0]) ** 2 / 7) * \
                         np.exp(-(spatial_coords[1] - center[1]) ** 2 / 7)

        # ---- apply stripe if needed ----
        if stripe and stripe_mask is not None:
            V_tmp[stripe_mask, :] = 0.0
            V_tmp[stripe_mask, stripe_type] = 1.0  # make pure chosen type

        # ---- renormalize ----
        V_tmp = V_tmp / np.mean(V_tmp, axis=1).reshape(-1, 1)
        V_true_temporal[i] = V_tmp

    # ---- build temporal synthetic AnnData list ----
    adata_synthetic_temp = [0] * len(t)
    for i in t:
        counts = V_true_temporal[i] @ pd_B.values + \
                 np.random.rand(V_true_temporal[i].shape[0], pd_B.shape[1]) * noise
        counts[counts < 0] = 0
        print(counts.shape)

        adata_synthetic_temp[i] = ad.AnnData(counts)
        adata_synthetic_temp[i].obs['x'] = adata.obs['x'].values
        adata_synthetic_temp[i].obs['y'] = adata.obs['y'].values
        adata_synthetic_temp[i].obs_names = [
            f"Cell_{i+1:d}" for i in range(adata_synthetic.n_obs)
        ]
        adata_synthetic_temp[i].var_names = [
            f"Gene {i+1:d}" for i in range(adata_synthetic.n_vars)
        ]

    return adata_synthetic_temp, V_true_temporal, adata, pd_B



import numpy as np
import pandas as pd
import anndata as ad

def create_circular_data(
    noise=2,
    n_time=1,
    seed=0,
    radius=25,
    vary_radius=False,          # NEW BOOLEAN
    sector_fracs=(0.5, 0.3, 0.2),
    sector_center=None,
):
    """
    Synthetic circular domain with uneven angular sectors and rotation across slices.
    Optionally grow radius linearly from (radius - 5) to (radius + 5).
    """
    rng = np.random.default_rng(seed)

    # ----------------------------------------------------------
    # Determine radii for each time point
    # ----------------------------------------------------------
    if vary_radius:
        if n_time > 1:
            radii = [round(
                (radius - 5) + 10 * (t / (n_time - 1)))
                for t in range(n_time)
            ]
        else:  # n_time == 1
            radii = [radius]
    else:
        radii = [radius] * n_time
    print(radii)

    # ----------------------------------------------------------
    # Angular partition fractions
    # ----------------------------------------------------------
    sector_fracs = np.asarray(sector_fracs, dtype=float)
    sector_fracs /= sector_fracs.sum()
    cum = 2.0 * np.pi * np.cumsum(sector_fracs)
    s1, s2, _ = cum

    # ----------------------------------------------------------
    # Cell-type-by-gene matrix B
    # ----------------------------------------------------------
    pd_B = pd.DataFrame(
        data=[[0.0, 1.0, 0.0, 0.1, 0.2],
              [1.0, 0.0, 0.0, 0.1, 0.0],
              [0.5, 0.0, 1.0, 0.1, 0.5]],
        columns=['Gene 1', 'Gene 2', 'Gene 3', 'Gene 4', 'Gene 5'],
        index=['Type 1', 'Type 2', 'Type 3']
    )

    V_true_temporal = [None] * n_time
    adata_synthetic_temp = [None] * n_time

    # ----------------------------------------------------------
    # Build each time slice
    # ----------------------------------------------------------
    for t in range(n_time):
        r_t = radii[t]

        # ------------------------------------------------------
        # 1. Build circular spatial template (triangular lattice)
        # ------------------------------------------------------
        xs = np.arange(-r_t, r_t + 1, 1)
        ys = np.arange(-r_t, r_t + 1, 1)
        X, Y = np.meshgrid(xs, ys)

        mask_circle = (X**2 + Y**2) <= r_t**2
        mask_tri = ((X + Y) % 2 == 1)
        mask = mask_circle & mask_tri

        x_coords = X[mask].ravel().astype(float)
        y_coords = Y[mask].ravel().astype(float)
        spatial_coords = np.vstack([x_coords, y_coords])
        n_spots = spatial_coords.shape[1]

        # ------------------------------------------------------
        # 2. Angle center
        # ------------------------------------------------------
        if sector_center is None:
            cx, cy = np.mean(spatial_coords, axis=1)
        else:
            cx, cy = sector_center

        base_angle = np.arctan2(
            spatial_coords[1] - cy,
            spatial_coords[0] - cx,
        )

        # ------------------------------------------------------
        # 3. Rotating sectors
        # ------------------------------------------------------
        if n_time > 1:
            theta = 2.0 * np.pi * t / n_time / 3
        else:
            theta = 0.0
        ang = np.mod(base_angle - theta, 2.0 * np.pi)

        # Assign types by sector
        type_idx = np.zeros(n_spots, dtype=int)
        type_idx[ang >= s1] = 1
        type_idx[ang >= s2] = 2

        V_tmp = np.zeros((n_spots, 3), float)
        V_tmp[np.arange(n_spots), type_idx] = 1.0
        V_true_temporal[t] = V_tmp

        # ------------------------------------------------------
        # 4. Build AnnData with gene expression
        # ------------------------------------------------------
        counts = V_tmp @ pd_B.values + rng.random((n_spots, pd_B.shape[1])) * noise
        counts[counts < 0] = 0

        adata_t = ad.AnnData(counts)
        adata_t.obs['x'] = x_coords
        adata_t.obs['y'] = y_coords
        adata_t.obs_names = [f"Cell_{k}" for k in range(n_spots)]
        adata_t.var_names = [f"Gene {k+1}" for k in range(pd_B.shape[1])]
        adata_synthetic_temp[t] = adata_t

    return adata_synthetic_temp, V_true_temporal, adata_synthetic_temp[0].copy(), pd_B
