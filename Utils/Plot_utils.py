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

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Wedge, Patch

def scatter_pies(
    ax,
    x,
    y,
    percents,
    radius=0.35,
    labels=None,
    colors=None,
    start_angle=90,
    edgecolor="white",
    lw=0.2,
    top_n=None,
    mode="full",   # "full" = pie chart, "argmax" = dominant type only
    leg_vis=True,
    keep_labels=None,        # NEW: list of labels to keep
    others_label="Others",   # NEW: name for merged remainder
    others_color=None,       # NEW: optional fixed color for Others
):
    """
    Visualize cell type composition per spot.

    NEW functionality:
    - keep_labels: keep only these cell types (by label name) and merge the rest into `others_label`.
      If keep_labels is None, behavior is unchanged (except top_n may still apply).
    """
    x = np.asarray(x)
    y = np.asarray(y)
    P = np.asarray(percents, float)
    P = np.nan_to_num(P, nan=0.0)

    n, k = P.shape
    if labels is None:
        labels = [f"C{i+1}" for i in range(k)]
    else:
        labels = list(labels)

    # ---------------------------------------------------------------
    # NEW: keep selected labels and merge the rest as "Others"
    # ---------------------------------------------------------------
    if keep_labels is not None:
        keep_labels = list(keep_labels)

        # map label -> index (first occurrence wins)
        label_to_idx = {lab: i for i, lab in enumerate(labels)}

        missing = [lab for lab in keep_labels if lab not in label_to_idx]
        if len(missing) > 0:
            raise ValueError(f"keep_labels contains labels not found in `labels`: {missing}")

        keep_idx = [label_to_idx[lab] for lab in keep_labels]
        drop_mask = np.ones(k, dtype=bool)
        drop_mask[keep_idx] = False
        drop_idx = np.where(drop_mask)[0]

        P_keep = P[:, keep_idx]
        P_others = P[:, drop_idx].sum(axis=1, keepdims=True) if drop_idx.size > 0 else np.zeros((n, 1))

        # Build new P/labels
        P = np.concatenate([P_keep, P_others], axis=1)
        labels = keep_labels + [others_label]
        k = P.shape[1]

        # If colors were provided, we subset and append Others
        if colors is not None:
            colors = list(colors)
            if len(colors) < max(keep_idx) + 1:
                raise ValueError(
                    "Provided `colors` is shorter than needed for the original `labels` indexing."
                )
            new_colors = [colors[i] for i in keep_idx]
            if others_color is None:
                # pick a deterministic extra color
                cmap = plt.get_cmap("tab20")
                others_color = cmap(len(new_colors) % 20)
            new_colors.append(others_color)
            colors = new_colors

    # ---------------------------------------------------------------
    # Row-normalize AFTER any merging/subsetting
    # ---------------------------------------------------------------
    rs = P.sum(axis=1, keepdims=True)
    rs[rs == 0] = 1.0
    P = P / rs

    # ---------------------------------------------------------------
    # Existing: keep only top_n cell types globally (by average abundance)
    # Note: if keep_labels is used, `labels` includes Others; top_n applies to this set.
    # ---------------------------------------------------------------
    keep_idx = np.arange(k)
    if top_n is not None and top_n < k:
        avg = P.mean(axis=0)
        keep_idx = np.argsort(avg)[::-1][:top_n]
        P = P[:, keep_idx]
        labels = [labels[i] for i in keep_idx]
        k = top_n

        if colors is not None and len(colors) >= (np.max(keep_idx) + 1):
            colors = [colors[i] for i in keep_idx]
        else:
            colors = None  # fallback to auto colors

    # Colors (auto if still None)
    if colors is None:
        cmap = plt.get_cmap("tab20")
        colors = [cmap(i % 20) for i in range(k)]
        # if we have Others and user requested a fixed Others color
        if others_color is not None and others_label in labels:
            colors[labels.index(others_label)] = others_color

    # ------------------------------------------------------------------
    # Draw
    # ------------------------------------------------------------------
    if mode == "full":
        for xi, yi, fracs in zip(x, y, P):
            theta = start_angle
            for f, c in zip(fracs, colors):
                if f <= 0:
                    continue
                w = Wedge(
                    (xi, yi),
                    r=radius,
                    theta1=theta,
                    theta2=theta + 360.0 * f,
                    facecolor=c,
                    edgecolor=edgecolor,
                    linewidth=lw,
                )
                ax.add_patch(w)
                theta += 360.0 * f

    elif mode == "argmax":
        max_idx = np.argmax(P, axis=1)
        for xi, yi, j in zip(x, y, max_idx):
            w = Wedge(
                (xi, yi),
                r=radius,
                theta1=0.0,
                theta2=360.0,
                facecolor=colors[j],
                edgecolor=edgecolor,
                linewidth=lw,
            )
            ax.add_patch(w)
    else:
        raise ValueError("mode must be 'full' or 'argmax'")

    # View
    ax.set_xlim(x.min() - radius * 1.1, x.max() + radius * 1.1)
    ax.set_ylim(y.min() - radius * 1.1, y.max() + radius * 1.1)
    ax.set_aspect("equal", adjustable="box")

    # Legend
    handles = [Patch(facecolor=colors[i], edgecolor=edgecolor, label=labels[i]) for i in range(k)]
    if leg_vis:
        ax.legend(
            handles=handles,
            title="Cell types",
            frameon=False,
            loc="center left",
            bbox_to_anchor=(1.02, 0.5),
        )
    return handles






import numpy as np
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

import anndata as ad
import scanpy as sc


def _run_graph_clustering(V, method="louvain", n_neighbors=15, resolution=1.0, random_state=0):
    tmp = ad.AnnData(X=np.asarray(V))
    sc.pp.neighbors(tmp, n_neighbors=n_neighbors, use_rep=None, random_state=random_state)

    key = "clusters"
    if method == "louvain":
        sc.tl.louvain(tmp, resolution=resolution, key_added=key, random_state=random_state)
    elif method == "leiden":
        sc.tl.leiden(tmp, resolution=resolution, key_added=key, random_state=random_state)
    else:
        raise ValueError("method must be 'louvain' or 'leiden'")

    labels = tmp.obs[key].astype("category").cat.codes.to_numpy()
    return labels, np.unique(labels).size

import numpy as np

def _find_resolution_for_target_clusters(
    V,
    method,
    target_n_clusters,
    n_neighbors=15,
    random_state=0,
    resolutions=None,
):
    """
    Internal helper: use only the interval ends determined by `resolutions`
    (either provided or constructed exactly as in the original code), then
    bisection search within [resolutions[0], resolutions[-1]].

    Returns (best_labels, best_resolution, best_n_clusters).
    """
    if resolutions is None:
        # --- your original grid logic (unchanged) ---
        resolutions = np.unique(np.r_[np.logspace(-2, 0, 20), np.logspace(0, 2, 50)])
        if target_n_clusters < 20:
            resolutions = np.linspace(0.01, 1.2, 100)
        elif target_n_clusters < 30:
            resolutions = np.linspace(0.5, 1.5, 100)
        else:
            resolutions = np.linspace(1.0, 3.0, 100)

    # Use ONLY the ends you chose via `resolutions`
    res_lo = float(resolutions[0])
    res_hi = float(resolutions[-1])

    def run(res):
        return _run_graph_clustering(
            V,
            method=method,
            n_neighbors=n_neighbors,
            resolution=res,
            random_state=random_state,
        )

    # Evaluate endpoints
    labels_lo, n_lo = run(res_lo)
    labels_hi, n_hi = run(res_hi)

    # Track best-so-far (same objective as your scan: closest to target)
    best_labels = labels_lo
    best_res = res_lo
    best_n_clust = n_lo
    best_diff = abs(n_lo - target_n_clusters)

    def update(labels, res, n_clust):
        nonlocal best_labels, best_res, best_n_clust, best_diff
        diff = abs(n_clust - target_n_clusters)
        if diff < best_diff:
            best_diff = diff
            best_labels = labels
            best_res = res
            best_n_clust = n_clust

    update(labels_hi, res_hi, n_hi)

    # If the target is not between endpoint cluster counts, bisection cannot help.
    if not (min(n_lo, n_hi) <= target_n_clusters <= max(n_lo, n_hi)):
        return best_labels, best_res, best_n_clust

    # Ensure we bisect in a direction where n_clust is increasing with res.
    # (If your method happens to produce n_lo > n_hi, swap ends.)
    if n_lo > n_hi:
        res_lo, res_hi = res_hi, res_lo
        n_lo, n_hi = n_hi, n_lo

    # Bisection
    max_iter = 25
    tol_res = 1e-3

    for _ in range(max_iter):
        res_mid = 0.5 * (res_lo + res_hi)
        labels_mid, n_mid = run(res_mid)
        update(labels_mid, res_mid, n_mid)

        if n_mid == target_n_clusters:
            return labels_mid, res_mid, n_mid

        if n_mid < target_n_clusters:
            res_lo, n_lo = res_mid, n_mid
        else:
            res_hi, n_hi = res_mid, n_mid

        if abs(res_hi - res_lo) < tol_res:
            break
        if n_lo == n_hi:
            break  # plateau/saturation

    return best_labels, best_res, best_n_clust


# def _find_resolution_for_target_clusters(
#     V,
#     method,
#     target_n_clusters,
#     n_neighbors=15,
#     random_state=0,
#     resolutions=None,
# ):
#     """
#     Internal helper: scan over 'resolutions' and pick the one whose
#     number of clusters is closest to target_n_clusters.

#     Returns (best_labels, best_resolution, best_n_clusters).
#     """
#     if resolutions is None:
#         # adjust grid as needed
#         # resolutions = np.linspace(0.2, 3.0, 15)
#         resolutions = np.unique(np.r_[np.logspace(-2, 0, 20), np.logspace(0, 2, 50)])
#         if target_n_clusters < 20:
#             resolutions = np.linspace(0.01,1.2,100)
#         elif target_n_clusters < 30:
#             resolutions = np.linspace(0.5,1.5,100)
#         else:
#             resolutions = np.linspace(1.0,3.0,100)
# # ~ [0.01 ... 10], denser at small values


#     best_labels = None
#     best_res = None
#     best_n_clust = None
#     best_diff = np.inf

#     for res in resolutions:
#         labels, n_clust = _run_graph_clustering(
#             V,
#             method=method,
#             n_neighbors=n_neighbors,
#             resolution=res,
#             random_state=random_state,
#         )
#         diff = abs(n_clust - target_n_clusters)
#         if diff < best_diff:
#             best_diff = diff
#             best_labels = labels
#             best_res = res
#             best_n_clust = n_clust

#         if diff == 0:
#             break

#     return best_labels, best_res, best_n_clust







def _find_resolution_for_best_ari(
    V,
    true_labels,
    method="louvain",
    n_neighbors=15,
    random_state=0,
    resolutions=None,
    slice_sizes=None,      # REQUIRED for per-slice objective
    objective="mean",      # "mean" or "median"
):
    """
    Choose resolution that maximizes mean/median per-slice ARI.

    Returns
    -------
    best_pred : (N,) np.ndarray
    best_res : float
    best_n : int
    best_score : float   (mean/median per-slice ARI)
    """
    if resolutions is None:
        resolutions = np.unique(np.r_[np.logspace(-2, 0, 10), np.logspace(0, 1, 20)])

    V = np.asarray(V)
    true_labels = np.asarray(true_labels)

    if slice_sizes is None:
        raise ValueError("slice_sizes must be provided to maximize per-slice ARI.")

    slice_sizes = list(map(int, slice_sizes))
    if sum(slice_sizes) != len(true_labels):
        raise ValueError(f"sum(slice_sizes)={sum(slice_sizes)} must equal len(true_labels)={len(true_labels)}")

    starts = np.cumsum([0] + slice_sizes[:-1])
    ends = np.cumsum(slice_sizes)

    if objective not in {"mean", "median"}:
        raise ValueError("objective must be 'mean' or 'median'")

    best_pred, best_res, best_n = None, None, None
    best_score = -np.inf

    for res in resolutions:
        pred, n_clust = _run_graph_clustering(
            V,
            method=method,
            n_neighbors=n_neighbors,
            resolution=float(res),
            random_state=random_state,
        )

        aris = [adjusted_rand_score(true_labels[st:en], pred[st:en]) for st, en in zip(starts, ends)]
        score = float(np.mean(aris) if objective == "mean" else np.median(aris))

        if score > best_score:
            best_score = score
            best_pred = pred
            best_res = float(res)
            best_n = int(n_clust)

    return best_pred, best_res, best_n, float(best_score)





def cluster_compute_ari_and_plot(
    V,
    adata,
    slice_sizes=None,
    celltype_key="cell_type",
    embedding_key="X_umap",
    clustering="louvain",
    n_clusters=None,
    n_neighbors=15,
    resolution=1.0,
    auto_match_n_clusters=True,
    resolution_grid=None,
    random_state=0,
    figsize=(12, 5),
    plot=True,
    auto_mode="max_ari",   # <--- NEW: "match_k" or "max_ari"
):
    V = np.asarray(V)
    true_labels = adata.obs[celltype_key].astype("category").cat.codes.to_numpy()

    if V.shape[0] != len(true_labels):
        raise ValueError("Row count in V must match number of cells in adata.")

    clustering = clustering.lower()

    if clustering == "kmeans":
        if n_clusters is None:
            n_clusters = np.unique(true_labels).size
        kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init="auto")
        pred_labels = kmeans.fit_predict(V)

    elif clustering in {"louvain", "leiden"}:
        method = clustering

        if auto_match_n_clusters:
            if auto_mode == "match_k":
                target_k = n_clusters if n_clusters is not None else np.unique(true_labels).size
                pred_labels, best_res, n_clust = _find_resolution_for_target_clusters(
                    V,
                    method=method,
                    target_n_clusters=target_k,
                    n_neighbors=n_neighbors,
                    random_state=random_state,
                    resolutions=resolution_grid,
                )
                print(f"{method.capitalize()}: target_k={target_k}, best_resolution={best_res:.3f}, n_clusters={n_clust}")

            elif auto_mode == "max_ari":
                
                pred_labels, best_res, n_clust, best_ari = _find_resolution_for_best_ari(
                    V,
                    slice_sizes = slice_sizes,
                    true_labels=true_labels,
                    method=method,
                    n_neighbors=n_neighbors,
                    random_state=random_state,
                    resolutions=resolution_grid,
                )
                print(f"{method.capitalize()}: best_resolution={best_res:.3f}, n_clusters={n_clust}, best_ARI={best_ari:.3f}")

            else:
                raise ValueError("auto_mode must be 'match_k' or 'max_ari'")

        else:
            pred_labels, n_clust = _run_graph_clustering(
                V,
                method=method,
                n_neighbors=n_neighbors,
                resolution=resolution,
                random_state=random_state,
            )
            print(f"{method.capitalize()}: resolution={resolution:.3f}, n_clusters={n_clust}")

    else:
        raise ValueError(f"Unknown clustering method: {clustering}")

    ari = adjusted_rand_score(true_labels, pred_labels)
    nmi = normalized_mutual_info_score(true_labels, pred_labels)

    # plotting unchanged
    emb = adata.obsm[embedding_key]
    x, y = emb[:, 0], emb[:, 1]

    if plot:
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        axes[0].scatter(x, y, c=true_labels, cmap="tab20", s=8)
        axes[0].set_title("True labels\n" + celltype_key)
        axes[1].scatter(x, y, c=pred_labels, cmap="tab20", s=8)
        axes[1].set_title(f"{clustering.capitalize()} clusters\n(ARI = {ari:.3f})")
        plt.tight_layout()
        plt.show()

    return ari, nmi, true_labels, pred_labels



import numpy as np
import matplotlib.pyplot as plt


def plot_cylinder_slices_with_couplings(
    adata_list,
    V_true_temporal,
    couplings,
    scatter_pies,
    t_start=0,
    t_end=None,        # inclusive
    min_weight=0.1,
    pie_radius=0.5,
    y_gap=60.0,
    y_scale=0.5,
    max_edges_per_pair=100,
    labels=None,
    colors=None,
    mode="full",
    top_n=None,
    figsize=(4, 6),
    leg_vis=True,
    # ---------------- NEW ----------------
    celltype_obs_key="cell_type",        # obs column name (used if edge_celltype_source="obs")
    edge_celltype_source="V_argmax",     # {"V_argmax", "obs"}
    edge_same_celltype=False,            # keep edge only if src and tgt have same type
    edge_celltype=None,                 # optional: restrict to one type (int index or str name)
):
    """
    Plot time slices as stacked discs (a "cylinder" view) with pie markers and coupling edges.

    Edge filtering options:
    - edge_celltype_source="V_argmax": types are inferred by argmax over rows of V_true_temporal[t]
      (indices correspond to columns of V / 'labels' if provided).
    - edge_celltype_source="obs": types are taken from adata.obs[celltype_obs_key] (categorical codes).

    If edge_same_celltype=True, draw an edge (i->j) only if type_t[i] == type_{t+1}[j].
    If edge_celltype is not None, additionally require type_t[i] == edge_celltype (and tgt matches if edge_same_celltype=True).

    Parameters
    ----------
    celltype_obs_key : str
        Column in adata.obs specifying cell type (used only if edge_celltype_source="obs").
    edge_celltype_source : str
        "V_argmax" or "obs"
    edge_celltype : int|str|None
        If int: interpreted as type index (V column index for V_argmax; category code for obs).
        If str: interpreted as a type name; mapped via `labels` (V_argmax) or obs categories (obs).
    """

    # ------------------------------
    # 1. Determine valid time window
    # ------------------------------
    T = len(adata_list)
    if t_end is None:
        t_end = T - 1
    if not (0 <= t_start <= t_end < T):
        raise ValueError("Invalid t_start, t_end range.")

    time_range = list(range(t_start, t_end + 1))
    n_time = len(time_range)

    adatas = [adata_list[t] for t in time_range]
    Vs = [np.asarray(V_true_temporal[t]) for t in time_range]

    # Slice couplings: if we include slices [s, ..., e], we need couplings [s->s+1, ..., e-1->e]
    sliced_couplings = []
    if n_time > 1:
        for t in range(t_start, t_end):
            sliced_couplings.append(couplings[t])

    # ----------------------------------------------------------
    # 2. Compute global center and transformed coords per slice
    # ----------------------------------------------------------
    all_xy_raw = np.vstack([
        ad.obs[["x", "y"]].to_numpy() - ad.obs[["x", "y"]].to_numpy().mean(axis=0, keepdims=True)
        for ad in adatas
    ])
    center = all_xy_raw.mean(axis=0)

    coords_transformed = []
    for idx, ad in enumerate(adatas):
        xy = ad.obs[["x", "y"]].to_numpy() - ad.obs[["x", "y"]].to_numpy().mean(axis=0, keepdims=True)
        xy_centered = xy - center
        xy_squash = np.column_stack([xy_centered[:, 0], y_scale * xy_centered[:, 1]])
        offset_y = ((n_time - 1) / 2.0 - idx) * y_gap
        xy_ellip = xy_squash + np.array([0.0, offset_y])
        coords_transformed.append(xy_ellip)

    # ----------------------------------------------------------
    # 3. Concatenate for single scatter_pies call
    # ----------------------------------------------------------
    x_all = np.concatenate([XY[:, 0] for XY in coords_transformed])
    y_all = np.concatenate([XY[:, 1] for XY in coords_transformed])
    P_all = np.concatenate(Vs, axis=0)

    fig, ax = plt.subplots(figsize=figsize)

    scatter_pies(
        ax=ax,
        x=x_all,
        y=y_all,
        percents=P_all,
        radius=pie_radius,
        labels=labels,
        colors=colors,
        start_angle=90,
        edgecolor="white",
        lw=0.2,
        top_n=top_n,
        mode=mode,
        leg_vis=leg_vis,
    )

    # ----------------------------------------------------------
    # 3.5 Compute per-slice type labels for edge filtering
    # ----------------------------------------------------------
    if edge_celltype_source not in {"V_argmax", "obs"}:
        raise ValueError("edge_celltype_source must be one of {'V_argmax','obs'}.")

    celltype_labels = []
    obs_categories = None

    if edge_celltype_source == "V_argmax":
        # hard type per point/spot = argmax over V row
        celltype_labels = [V.argmax(axis=1) for V in Vs]
    else:
        # hard type per point/spot = category code from adata.obs[celltype_obs_key]
        if celltype_obs_key is None:
            raise ValueError("celltype_obs_key must be provided when edge_celltype_source='obs'.")
        for ad in adatas:
            ct = ad.obs[celltype_obs_key].astype("category")
            celltype_labels.append(ct.cat.codes.to_numpy())
        obs_categories = adatas[0].obs[celltype_obs_key].astype("category").cat.categories

    # Map edge_celltype (optional) to an integer index/code
    if isinstance(edge_celltype, str):
        if edge_celltype_source == "V_argmax":
            if labels is None:
                raise ValueError("If edge_celltype is a string and edge_celltype_source='V_argmax', provide `labels`.")
            if edge_celltype not in list(labels):
                raise ValueError(f"edge_celltype='{edge_celltype}' not found in labels.")
            edge_celltype_idx = int(list(labels).index(edge_celltype))
        else:
            if obs_categories is None:
                raise RuntimeError("obs_categories unavailable (unexpected).")
            if edge_celltype not in list(obs_categories):
                raise ValueError(f"edge_celltype='{edge_celltype}' not found in adata.obs['{celltype_obs_key}'] categories.")
            edge_celltype_idx = int(list(obs_categories).index(edge_celltype))
    elif edge_celltype is None:
        edge_celltype_idx = None
    else:
        edge_celltype_idx = int(edge_celltype)

    # ----------------------------------------------------------
    # 4. Draw couplings between consecutive slices (with filtering)
    # ----------------------------------------------------------
    for k in range(n_time - 1):
        Traw = np.asarray(sliced_couplings[k])
        T12 = (Traw / Traw.max()) if (Traw.max() > 0) else Traw

        XY_t = coords_transformed[k]
        XY_tp = coords_transformed[k + 1]

        n_t, n_tp = T12.shape
        if XY_t.shape[0] != n_t or XY_tp.shape[0] != n_tp:
            raise ValueError("Coupling matrix shape does not match number of points in slices.")

        # candidate edges by weight
        i_idx, j_idx = np.where(T12 >= min_weight)
        if i_idx.size == 0:
            continue

        # -------- NEW: filter by cell types --------
        if edge_same_celltype or (edge_celltype_idx is not None):
            lt = celltype_labels[k]
            ltp = celltype_labels[k + 1]

            keep = np.ones_like(i_idx, dtype=bool)

            if edge_same_celltype:
                keep &= (lt[i_idx] == ltp[j_idx])

            if edge_celltype_idx is not None:
                keep &= (lt[i_idx] == edge_celltype_idx)

            i_idx, j_idx = i_idx[keep], j_idx[keep]
            if i_idx.size == 0:
                continue

        w_vec = T12[i_idx, j_idx]

        # strongest edges first
        order = np.argsort(w_vec)[::-1]
        if max_edges_per_pair is not None:
            order = order[:max_edges_per_pair]

        i_sel = i_idx[order]
        j_sel = j_idx[order]
        w_sel = w_vec[order]

        for i, j, w in zip(i_sel, j_sel, w_sel):
            x1, y1 = XY_t[i]
            x2, y2 = XY_tp[j]
            ax.plot(
                [x1, x2],
                [y1, y2],
                color=(0.1, 0.1, 0.1),
                alpha=0.1,
                linewidth=2.,
                zorder=1,
            )

    # ----------------------------------------------------------
    # 5. Final plot formatting
    # ----------------------------------------------------------
    pad = pie_radius * 1.5
    ax.set_xlim(x_all.min() - pad, x_all.max() + pad)
    ax.set_ylim(y_all.min() - pad, y_all.max() + pad)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    plt.tight_layout()

    return fig, ax



import numpy as np
import pandas as pd

def build_V_true_temporal_from_annotation(
    adata_list,
    ref_celltypes,
    celltype_key="annotation",
    unknown_label="Others",
    return_sparse=False,
    dtype=np.int8,
):
    """
    Build a 'V_true_temporal' list from a list of AnnData objects by converting
    per-cell annotations into one-hot (binary) matrices aligned to a reference
    cell-type ordering.

    Parameters
    ----------
    adata_list : list
        List of AnnData objects, one per time point (or slice group).
    ref_celltypes : list-like
        Reference ordering of cell types (columns) to use for all matrices.
        Any annotation not in this list is mapped to `unknown_label`.
    celltype_key : str
        Key in adata.obs containing the categorical cell-type annotation.
    unknown_label : str
        Label to use for annotations not present in `ref_celltypes`.
    return_sparse : bool
        If True, return scipy.sparse.csr_matrix for each time point.
    dtype : numpy dtype
        dtype for the returned matrices (default int8).

    Returns
    -------
    V_true_temporal : list
        List of length len(adata_list). Each item is an (n_cells_t x K) binary
        matrix aligned to the reference ordering (plus possibly `unknown_label`).
    celltypes_out : list
        The final cell-type ordering used (list of length K).
    """
    # Normalize ref order
    if isinstance(ref_celltypes, (pd.Index, pd.Series)):
        ref_celltypes = list(ref_celltypes)
    else:
        ref_celltypes = list(ref_celltypes)

    if len(ref_celltypes) == 0:
        raise ValueError("ref_celltypes must be a non-empty list of cell type names.")

    # Ensure unknown_label exists
    if unknown_label is not None and unknown_label not in ref_celltypes:
        celltypes_out = ref_celltypes + [unknown_label]
    else:
        celltypes_out = ref_celltypes

    K = len(celltypes_out)
    col_index = {ct: j for j, ct in enumerate(celltypes_out)}

    if return_sparse:
        from scipy import sparse

    V_true_temporal = []
    for t, adata in enumerate(adata_list):
        if celltype_key not in adata.obs:
            raise KeyError(
                f"adata_list[{t}] is missing obs['{celltype_key}']. "
                f"Available keys: {list(adata.obs_keys())}"
            )

        ct = adata.obs[celltype_key].astype(str).to_numpy()
        n = ct.shape[0]

        if return_sparse:
            rows = np.arange(n, dtype=np.int64)
            cols = np.empty(n, dtype=np.int64)
            for i, lab in enumerate(ct):
                if lab in col_index:
                    cols[i] = col_index[lab]
                elif unknown_label is not None:
                    cols[i] = col_index[unknown_label]
                else:
                    cols[i] = -1
            keep = cols >= 0
            data = np.ones(keep.sum(), dtype=dtype)
            M = sparse.csr_matrix(
                (data, (rows[keep], cols[keep])),
                shape=(n, K),
                dtype=dtype,
            )
        else:
            M = np.zeros((n, K), dtype=dtype)
            for i, lab in enumerate(ct):
                if lab in col_index:
                    M[i, col_index[lab]] = 1
                elif unknown_label is not None:
                    M[i, col_index[unknown_label]] = 1

        V_true_temporal.append(M)

    return V_true_temporal, celltypes_out



import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def compute_transition_tables_from_couplings(
    adata_list,
    couplings,
    celltype_key="annotation",
    keep_labels=None,         # e.g. ["RGC","NeuB","GlioB"]
    others_label="Others",    # collapsed remainder label
    normalize="row",          # "row", "col", "total", or None
):
    """
    Aggregate cell-level couplings into cell-type transition tables.

    Parameters
    ----------
    adata_list : list of AnnData
        Length T. adata_list[t].obs[celltype_key] contains cell-type labels at time t.
    couplings : list of array-like
        Length T-1. couplings[t] is the coupling matrix between time t and t+1,
        shape (n_t, n_{t+1}).
    celltype_key : str
        obs column name with categorical labels.
    keep_labels : list of str or None
        If provided, keep only these labels and collapse all others into `others_label`.
    others_label : str or None
        Label name used for collapsed types. If None, unknown labels are dropped (rows become all-zero).
    normalize : {"row","col","total",None}
        Normalization for each transition table:
        - "row": each row sums to 1 (P(target type | source type))
        - "col": each column sums to 1
        - "total": whole table sums to 1
        - None: raw transported mass

    Returns
    -------
    tables : list of pd.DataFrame
        Length T-1. Each is (K x K) transition table from time t to t+1.
    labels_out : list of str
        The label ordering used for rows/cols.
    """
    # ---- small helper: collapse labels ----
    def _collapse(arr):
        arr = pd.Series(arr, dtype="string")
        if keep_labels is None:
            return arr.to_numpy()
        keep = set(keep_labels)
        if others_label is None:
            # drop unknowns by setting them to NA; later they contribute nothing
            arr = arr.where(arr.isin(keep), other=pd.NA)
        else:
            arr = arr.where(arr.isin(keep), other=others_label)
        return arr.to_numpy()

    T = len(adata_list)
    if len(couplings) != T - 1:
        raise ValueError(f"Expected len(couplings)=T-1={T-1}, got {len(couplings)}")

    # ---- collect labels per time point (collapsed if requested) ----
    labs_per_t = []
    for t, ad in enumerate(adata_list):
        if celltype_key not in ad.obs:
            raise KeyError(f"adata_list[{t}] missing obs['{celltype_key}']")
        labs = ad.obs[celltype_key].astype(str).to_numpy()
        labs_per_t.append(_collapse(labs))

    # ---- build global label ordering ----
    if keep_labels is not None:
        labels_out = list(keep_labels)
        if others_label is not None:
            # add Others only if it appears
            if any(np.any(l == others_label) for l in labs_per_t):
                if others_label not in labels_out:
                    labels_out.append(others_label)
        # if others_label is None, unknowns are NA and won't appear
    else:
        labels_out = sorted(set(np.concatenate([l[~pd.isna(l)] for l in labs_per_t])))

    K = len(labels_out)
    lab_to_idx = {lab: i for i, lab in enumerate(labels_out)}

    # ---- sparse support (optional) ----
    try:
        from scipy import sparse
        _has_sparse = True
    except Exception:
        _has_sparse = False

    tables = []
    for t in range(T - 1):
        G = couplings[t]
        n0 = adata_list[t].n_obs
        n1 = adata_list[t + 1].n_obs

        if getattr(G, "shape", None) != (n0, n1):
            raise ValueError(
                f"couplings[{t}] has shape {getattr(G,'shape',None)}, expected {(n0, n1)}"
            )

        src = labs_per_t[t]
        tgt = labs_per_t[t + 1]

        M = np.zeros((K, K), dtype=float)

        is_sparse = _has_sparse and sparse.issparse(G)
        if is_sparse:
            G = G.tocoo()
            for i, j, v in zip(G.row, G.col, G.data):
                si = src[i]
                tj = tgt[j]
                if pd.isna(si) or pd.isna(tj):
                    continue
                a = lab_to_idx[si]
                b = lab_to_idx[tj]
                M[a, b] += float(v)
        else:
            # dense: accumulate by source label blocks
            tgt_idx = np.array([-1 if pd.isna(l) else lab_to_idx[l] for l in tgt], dtype=int)
            for a_lab, a in lab_to_idx.items():
                src_mask = (src == a_lab)
                if not np.any(src_mask):
                    continue
                block = np.asarray(G[src_mask, :])
                s = block.sum(axis=0).ravel()  # mass to each target cell
                good = tgt_idx >= 0
                np.add.at(M[a, :], tgt_idx[good], s[good])

        # ---- normalize if requested ----
        if normalize is None:
            Mn = M
        elif normalize == "row":
            rs = M.sum(axis=1, keepdims=True)
            rs[rs == 0] = 1.0
            Mn = M / rs
        elif normalize == "col":
            cs = M.sum(axis=0, keepdims=True)
            cs[cs == 0] = 1.0
            Mn = M / cs
        elif normalize == "total":
            tot = M.sum()
            Mn = M / (tot if tot != 0 else 1.0)
        else:
            raise ValueError("normalize must be one of {'row','col','total',None}")

        tables.append(pd.DataFrame(Mn, index=labels_out, columns=labels_out))

    return tables, labels_out





import numpy as np
from collections import Counter, defaultdict

def _majority_mapping(pred_labels, true_labels):
    """
    Map each cluster id -> majority true label (in code space).
    Returns dict[int,int].
    """
    mapping = {}
    for c in np.unique(pred_labels):
        idx = (pred_labels == c)
        if idx.sum() == 0:
            continue
        # majority vote among true labels in this cluster
        mapping[c] = Counter(true_labels[idx]).most_common(1)[0][0]
    return mapping

def _apply_mapping(pred_labels, mapping, default=-1):
    return np.array([mapping.get(c, default) for c in pred_labels], dtype=int)

def _accuracy(y_true, y_pred):
    mask = (y_pred >= 0)
    if mask.sum() == 0:
        return np.nan
    return (y_true[mask] == y_pred[mask]).mean()

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import adjusted_rand_score



def global_cluster_then_per_slice_ari(
    V_list,
    adata_list,
    celltype_key="bin_annotation",
    embedding_key="spatial",
    clustering="louvain",
    n_clusters=None,
    n_neighbors=15,
    resolution=1.0,
    auto_match_n_clusters=True,
    resolution_grid=None,
    random_state=0,
    plot=False,                 # <---- toggle here
    point_size=8,
    cmap="tab20",
    figsize=(10, 4),
    auto_mode='max_ari',
    regroup=False,
    sc_order = None,
    return_pred=False
):
    """
    Global clustering, then per-slice ARI evaluation.
    Optional per-slice plotting.
    """

    # Concatenate
    V_all = np.vstack(V_list)
    
    adata_all = ad.concat(
        adata_list,
        join="outer",
        label="slice_id",
        keys=[str(i) for i in range(len(adata_list))],
    )
    if regroup:
        from zesta_utils import aggregate_to_bins, sc_to_bin
        sc_dict = sc_to_bin()
        
        
        V_all,_ = aggregate_to_bins(V_all.copy(), sc_order, sc_dict)#, bin_order = adata_all.obs['bin_annotation'].unique().tolist())
    slice_sizes = [a.n_obs for a in adata_list]
    # Global clustering
    global_ari, global_nmi, true_all, pred_all = cluster_compute_ari_and_plot(
        V_all,
        adata_all,
        celltype_key=celltype_key,
        embedding_key=embedding_key,
        clustering=clustering,
        n_clusters=n_clusters,
        n_neighbors=n_neighbors,
        slice_sizes=slice_sizes,
        resolution=resolution,
        auto_match_n_clusters=auto_match_n_clusters,
        resolution_grid=resolution_grid,
        random_state=random_state,
        plot=False,   # <-- always False here
        auto_mode=auto_mode
    )

    # Per-slice ARI
    sizes = [a.n_obs for a in adata_list]
    starts = np.cumsum([0] + sizes[:-1])
    ends = np.cumsum(sizes)

    per_slice_ari = {}
    per_slice_nmi = {}
    pred_i_slice = {}

    for i, (st, en) in enumerate(zip(starts, ends)):
        true_i = true_all[st:en]
        pred_i = pred_all[st:en]
        pred_i_slice[i] = pred_i
        ari_i = adjusted_rand_score(true_i, pred_i)
        nmi_i = normalized_mutual_info_score(true_i, pred_i)
        per_slice_ari[i] = ari_i
        per_slice_nmi[i] = nmi_i

        if plot:
            adata_i = adata_list[i]
            emb = adata_i.obsm[embedding_key]
            x, y = emb[:, 0], emb[:, 1]

            fig, axes = plt.subplots(1, 2, figsize=figsize)

            axes[0].scatter(x, y, c=true_i, cmap=cmap, s=point_size)
            axes[0].set_title(f"Slice {i}: True")

            axes[1].scatter(x, y, c=pred_i, cmap=cmap, s=point_size)
            axes[1].set_title(f"Slice {i}: Global clusters\nARI={ari_i:.3f}")

            plt.tight_layout()
            plt.show()

    ari_vals = np.array(list(per_slice_ari.values()))
    nmi_vals = np.array(list(per_slice_nmi.values()))

    if return_pred == False:

        return {
            "global_ari": global_ari,
            "per_slice_ari": per_slice_ari,
            "global_nmi": global_nmi,
            "per_slice_nmi": per_slice_nmi,
            "median_ari": np.median(ari_vals),
            "median_nmi": np.median(nmi_vals),
            "mean_ari": np.mean(ari_vals),
            "mean_nmi": np.mean(nmi_vals),
            "weighted_mean_ari": np.average(ari_vals, weights=sizes),
        }
    else:
        return  {
            "global_ari": global_ari,
            "per_slice_ari": per_slice_ari,
            "global_nmi": global_nmi,
            "per_slice_nmi": per_slice_nmi,
            "median_ari": np.median(ari_vals),
            "median_nmi": np.median(nmi_vals),
            "mean_ari": np.mean(ari_vals),
            "mean_nmi": np.mean(nmi_vals),
            "weighted_mean_ari": np.average(ari_vals, weights=sizes),
            "pred_labels":pred_i_slice
        }


import numpy as np
import matplotlib.pyplot as plt

def scatter_sorted(
    x,
    y,
    c,
    ax=None,
    s=20,
    cmap="viridis",
    alpha=1.0,
    **kwargs,
):
    """
    Scatter plot where points are drawn in descending order of color values.

    Parameters
    ----------
    x, y : array-like, shape (n,)
        Coordinates.
    c : array-like, shape (n,)
        Values used for coloring (sorted descending).
    ax : matplotlib.axes.Axes or None
        Axis to plot on. If None, uses current axis.
    s : float
        Marker size.
    cmap : str or Colormap
        Colormap.
    alpha : float
        Marker transparency.
    **kwargs :
        Passed to plt.scatter.

    Returns
    -------
    sc : PathCollection
        The scatter object.
    """
    x = np.asarray(x)
    y = np.asarray(y)
    c = np.asarray(c, dtype=float)

    order = np.argsort(c)  # ascending: low first, high last (on top)

    if ax is None:
        ax = plt.gca()

    sc = ax.scatter(
        x[order], y[order],
        c=c[order],
        s=s, cmap=cmap, alpha=alpha,
        **kwargs
    )
    return sc

import numpy as np
import matplotlib.pyplot as plt


def _row_stochastic_from_pi(pi, eps=1e-12):
    pi = np.asarray(pi, dtype=float)
    den = pi.sum(axis=1, keepdims=True)
    den = np.maximum(den, eps)
    return pi / den


def _get_xy(adata, x_key="x", y_key="y", spatial_key="spatial"):
    if x_key in adata.obs and y_key in adata.obs:
        return np.asarray(adata.obs[x_key].values, float), np.asarray(adata.obs[y_key].values, float)
    if spatial_key in adata.obsm:
        xy = np.asarray(adata.obsm[spatial_key], float)
        return xy[:, 0], xy[:, 1]
    raise KeyError(f"Need obs['{x_key}'], obs['{y_key}'] or obsm['{spatial_key}'].")


def _sorted_scatter(ax, x, y, c, s=18, cmap="inferno", alpha=0.95, vmin=None, vmax=None, **kwargs):
    """
    Draw points so that high values appear on top:
    sort ascending (low first, high last).
    """
    x = np.asarray(x); y = np.asarray(y); c = np.asarray(c, float)
    order = np.argsort(c)  # low first, high last
    sc = ax.scatter(
        x[order], y[order],
        c=c[order],
        s=s,
        cmap=cmap,
        alpha=alpha,
        vmin=vmin,
        vmax=vmax,
        linewidths=0,
        **kwargs,
    )
    return sc

import numpy as np

def build_centered_stacked_coords(adatas, y_gap=500, y_scale=0.3, x_key="x", y_key="y", shift_dir='y'):
    """
    Implements exactly the coordinate transform you posted.

    Returns
    -------
    coords_transformed : list[np.ndarray]
        List of XY arrays, one per time, each of shape (n_spots_t, 2).
    x_all, y_all : np.ndarray
        Concatenated coordinates across all times.
    """
    n_time = len(adatas)

    all_xy_raw = np.vstack([
        ad.obs[[x_key, y_key]].to_numpy()
        - ad.obs[[x_key, y_key]].to_numpy().mean(axis=0, keepdims=True)
        for ad in adatas
    ])
    center = all_xy_raw.mean(axis=0)

    coords_transformed = []
    for idx, ad in enumerate(adatas):
        xy = ad.obs[[x_key, y_key]].to_numpy() - ad.obs[[x_key, y_key]].to_numpy().mean(axis=0, keepdims=True)
        xy_centered = xy - center
        xy_squash = np.column_stack([xy_centered[:, 0], y_scale * xy_centered[:, 1]])
        offset_y = ((n_time - 1) / 2.0 - idx) * y_gap
        if shift_dir == 'y':
            xy_ellip = xy_squash + np.array([0.0, offset_y])
        else:
            xy_ellip = xy_squash + np.array([-offset_y,0.0])
        coords_transformed.append(xy_ellip)

    x_all = np.concatenate([XY[:, 0] for XY in coords_transformed])
    y_all = np.concatenate([XY[:, 1] for XY in coords_transformed])
    return coords_transformed, x_all, y_all





def _sorted_scatter(ax, x, y, c, s=18, cmap="inferno", alpha=0.95, vmin=None, vmax=None, **kwargs):
    # low first, high last => hotspots drawn on top
    x = np.asarray(x); y = np.asarray(y); c = np.asarray(c, float)
    order = np.argsort(c)
    sc = ax.scatter(
        x[order], y[order],
        c=c[order],
        s=s,
        cmap=cmap,
        alpha=alpha,
        vmin=vmin,
        vmax=vmax,
        linewidths=0,
        **kwargs
    )
    return sc


def plot_mass_transport_heatmap(
    adata_insamp,
    V,
    pi,
    t_start=0,
    t_end=None,
    y_gap=500,
    y_scale=0.3,
    figsize=(8, 7),
    ax=None,
    fig=None,
    tight_layout=True,
    # source mass
    source_celltype=None,
    celltype_obs_key="bin_annotation",
    source_mass_source="obs",      # {"obs","V_argmax"}
    labels=None,                   # list of V column names if using V_argmax with str
    unit_mass="per_spot",          # {"per_spot","probability"}
    # edges (barycentric projection)
    show_edges=True,
    n_sources_show=300,            # number of source points to draw arrows from (None -> all)
    sources_rank_by="mass",        # {"mass","peakedness"} for choosing which sources to draw
    edge_alpha=0.35,
    edge_lw=0.6,
    edge_scale=1.0,                # multiply arrow displacement
    edge_same_celltype=False,      # keep only arrows whose source and target-expected celltype match
    edge_celltype_source="obs",    # {"obs","V_argmax"}
    edge_celltype=None,            # if set, only draw arrows whose source celltype == this
    seed=0,                        # used only for tie-breaking in ranking
    # heatmap style
    cmap="inferno",
    s=18,
    alpha=0.95,
    vmin=0.0,
    vmax=None,
    show_colorbar=True,
    title=None,
    # coords keys
    x_key="x",
    y_key="y",
    eps=1e-12,
):
    """
    Plot stacked slices with a transported-mass heatmap and (optionally) barycentric-projection edges.

    Heatmap: colors by transported mass m_t (seeded at t_start) propagated via row-stochastic P from pi.
    Edges: for each source i at time t, draw an arrow to the barycentric projection:
              y_bar(i) = sum_j P_ij * y_j
          using the *transformed/stacked* coordinates (so arrows align with the plot).

    Requires (already in your codebase):
      - build_centered_stacked_coords(adatas, y_gap, y_scale, x_key, y_key, shift_dir)
      - _row_stochastic_from_pi(Pi, eps)
    """
    rng = np.random.default_rng(seed)

    if source_celltype is None:
        raise ValueError("Pass source_celltype (e.g. 'Somite') to seed mass at t_start.")

    T = len(adata_insamp)
    if t_end is None:
        t_end = T - 1
    if not (0 <= t_start <= t_end < T):
        raise ValueError(f"Need 0 <= t_start <= t_end < {T}.")

    # Subset time window
    adatas = [adata_insamp[t] for t in range(t_start, t_end + 1)]
    Vs = [np.asarray(V[t]) for t in range(t_start, t_end + 1)]

    # ---- seed mass at first slice in this window (idx=0 corresponds to t_start)
    ad0 = adatas[0]
    V0 = Vs[0]

    if source_mass_source == "obs":
        if celltype_obs_key not in ad0.obs:
            raise KeyError(f"'{celltype_obs_key}' not found in adata_insamp[{t_start}].obs")
        obs0 = np.asarray(ad0.obs[celltype_obs_key].values)
        mask0 = (obs0 == source_celltype)

    elif source_mass_source == "V_argmax":
        types0 = V0.argmax(axis=1)
        if isinstance(source_celltype, str):
            if labels is None:
                raise ValueError("source_celltype is str with source_mass_source='V_argmax' but labels=None.")
            if source_celltype not in labels:
                raise ValueError(f"source_celltype='{source_celltype}' not in labels.")
            src_id = labels.index(source_celltype)
        else:
            src_id = int(source_celltype)
        mask0 = (types0 == src_id)

    else:
        raise ValueError("source_mass_source must be 'obs' or 'V_argmax'.")

    if mask0.sum() == 0:
        raise ValueError(f"No spots matched source_celltype='{source_celltype}' at t_start={t_start}.")

    m = np.zeros(ad0.n_obs, dtype=float)
    m[mask0] = 1.0
    if unit_mass == "probability":
        m = m / np.maximum(m.sum(), eps)
    elif unit_mass != "per_spot":
        raise ValueError("unit_mass must be 'per_spot' or 'probability'.")

    masses = [m]

    # ---- propagate within window using global pi indices (row-stochastic P)
    for t in range(t_start, t_end):
        Pi = np.asarray(pi[t], dtype=float)
        Pi_sum = Pi.sum()
        if Pi_sum <= eps:
            raise ValueError(f"pi[{t}] has non-positive total mass.")
        Pi = Pi / Pi_sum

        n_src = adata_insamp[t].n_obs
        n_tgt = adata_insamp[t + 1].n_obs
        if Pi.shape != (n_src, n_tgt):
            raise ValueError(f"pi[{t}] has shape {Pi.shape}, expected {(n_src, n_tgt)}.")

        P = _row_stochastic_from_pi(Pi, eps=eps)
        masses.append(np.asarray(masses[-1] @ P).ravel())

    # ---- coordinate transform + concatenation
    coords_transformed, x_all, y_all = build_centered_stacked_coords(
        adatas, y_gap=y_gap, y_scale=y_scale, x_key=x_key, y_key=y_key, shift_dir="x"
    )

    # ---- concatenate masses for single scatter call
    m_all = np.concatenate(masses, axis=0)
    if vmax is None:
        vmax = float(np.max(m_all)) if np.max(m_all) > 0 else 1.0

    # --- figure/axes handling
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        if fig is None:
            fig = ax.figure

    sc = _sorted_scatter(
        x=x_all,
        y=y_all,
        c=m_all,
        s=s,
        ax=ax,
        cmap=cmap,
        alpha=alpha,
        vmin=vmin,
        vmax=vmax,
        # linewidths=0,
    )

    # label each slice
    for idx, XY in enumerate(coords_transformed):
        ax.text(
            float(np.min(XY[:, 0])),
            float(np.min(XY[:, 1])) - 0.05 * y_gap,
            f"t={t_start + idx}",
            fontsize=10,
            ha="left",
            va="top",
        )

    # ---- barycentric-projection edges
    if show_edges:
        def _types_for_edge_filter(global_t):
            if edge_celltype_source == "V_argmax":
                return np.asarray(V[global_t]).argmax(axis=1)
            if edge_celltype_source == "obs":
                return np.asarray(adata_insamp[global_t].obs[celltype_obs_key].values)
            raise ValueError("edge_celltype_source must be 'obs' or 'V_argmax'.")

        edge_type = edge_celltype
        if edge_celltype_source == "V_argmax" and isinstance(edge_celltype, str):
            if labels is None:
                raise ValueError("edge_celltype is str with edge_celltype_source='V_argmax' but labels=None.")
            if edge_celltype not in labels:
                raise ValueError(f"edge_celltype='{edge_celltype}' not in labels.")
            edge_type = labels.index(edge_celltype)

        for local_idx, global_t in enumerate(range(t_start, t_end)):
            # transformed coordinates for this displayed pair
            XY0 = coords_transformed[local_idx]       # (n_src, 2) in displayed coordinates
            XY1 = coords_transformed[local_idx + 1]   # (n_tgt, 2) in displayed coordinates

            # build row-stochastic P for this transition
            Pi = np.asarray(pi[global_t], dtype=float)
            Pi_sum = Pi.sum()
            if Pi_sum <= eps:
                continue
            Pi = Pi / Pi_sum
            P = _row_stochastic_from_pi(Pi, eps=eps)

            n_src, n_tgt = P.shape
            if XY0.shape[0] != n_src or XY1.shape[0] != n_tgt:
                raise ValueError(
                    f"Coordinate mismatch at t={global_t}: "
                    f"P={P.shape}, XY0={XY0.shape}, XY1={XY1.shape}"
                )

            m_t = np.asarray(masses[local_idx]).ravel()
            if m_t.size != n_src:
                raise ValueError("Mass vector size mismatch with P rows.")

            src_types = _types_for_edge_filter(global_t)
            tgt_types = _types_for_edge_filter(global_t + 1)

            # choose candidate sources
            src_all = np.arange(n_src)

            # optional: only show sources of a specific type
            if edge_type is not None:
                src_all = src_all[src_types == edge_type]
            if src_all.size == 0:
                continue

            # rank sources for visualization
            if sources_rank_by == "mass":
                scores = m_t[src_all]
            elif sources_rank_by == "peakedness":
                scores = P[src_all].max(axis=1)
            else:
                raise ValueError("sources_rank_by must be 'mass' or 'peakedness'.")

            if n_sources_show is not None:
                n_show = int(n_sources_show)
                n_show = max(1, min(n_show, src_all.size))
                # break ties deterministically but non-pathologically
                jitter = 1e-15 * rng.random(src_all.size)
                pick = np.argpartition(scores + jitter, -n_show)[-n_show:]
                src_sel = src_all[pick]
            else:
                src_sel = src_all

            # barycentric target coordinates for selected sources:
            # y_bar(i) = sum_j P_ij * XY1[j]
            # shape: (len(src_sel), 2)
            Ybar = P[src_sel] @ XY1

            # optionally, barycentric "type" (argmax of expected target type)
            if edge_same_celltype:
                # need target types to be numeric to compare; if obs strings, compare strings
                # compute a soft vote for each target-type label:
                # simplest: take argmax target index (still okay here for type only)
                j_star = P[src_sel].argmax(axis=1)
                tgt_type_bar = tgt_types[j_star]
            else:
                tgt_type_bar = None

            for idx_i, i in enumerate(src_sel):
                if edge_same_celltype:
                    if src_types[i] != tgt_type_bar[idx_i]:
                        continue

                x0, y0 = XY0[i, 0], XY0[i, 1]
                x1, y1 = Ybar[idx_i, 0], Ybar[idx_i, 1]

                dx = (x1 - x0) * edge_scale
                dy = (y1 - y0) * edge_scale

                # draw as an arrow (quiver-like using annotate for consistent style)
                ax.annotate(
                    "",
                    xy=(x0 + dx, y0 + dy),
                    xytext=(x0, y0),
                    arrowprops=dict(arrowstyle="-|>", lw=edge_lw, alpha=edge_alpha, color="gray"),
                )

    ax.set_aspect("equal")
    ax.axis("off")
    if title is None:
        title = f"Mass transport heatmap from '{source_celltype}' (t={t_start})"
    ax.set_title(title)

    if show_colorbar:
        cbar = fig.colorbar(sc, ax=ax, fraction=0.03, pad=0.01)
        cbar.set_label("Transported mass")

    if tight_layout and (ax is None):
        plt.tight_layout()

    return fig, ax, masses



import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment

def plot_slices_horizontal_centered(
    adata_list,
    obs_key,
    coord_keys=("x", "y"),
    obsm_key="spatial",
    order_key=None,
    order_map=None,
    x_gap=None,
    y_scale=1.0,
    s=6,
    alpha=0.9,
    figsize=None,
    legend=True,
    legend_ncol=None,
    legend_loc="center left",
    legend_bbox=(1.02, 0.5),
    legend_font=7,
    show_ari=None,
    # --- NEW ---
    truth_key=None,
    align_colors=True,
    truth_cmap="Set3",
    unmatched_cmap="tab20c",
    return_alignment=False,
):
    """
    Plot multiple spatial slices in one scatter plot, centered per-slice at (0,0) and arranged
    left-to-right by adding an x-shift to each slice.

    NEW:
      - truth_key + align_colors=True aligns cluster labels (obs_key) to truth labels (truth_key)
        via Hungarian assignment on the global contingency table aggregated across slices.
      - Colors are assigned by truth label palette; each cluster inherits its matched truth color.
      - cat_order is ordered by truth label order (then unmatched clusters at end).

    Parameters (new)
    ----------------
    truth_key : str or None
        adata.obs column containing "ground truth" labels for alignment.
    align_colors : bool
        If True and truth_key is provided, align cluster colors/order to truth.
    truth_cmap : str
        Matplotlib colormap name for truth labels (muted palettes like "Set3" recommended).
    unmatched_cmap : str
        Colormap for clusters that cannot be matched (when #clusters > #truth labels).
    return_alignment : bool
        If True, return (fig, ax, alignment_dict, contingency_df).

    Returns
    -------
    fig, ax  (and optionally alignment outputs)
    """

    # -------------------------
    # helper: extract coords
    # -------------------------
    def _get_xy(ad):
        xk, yk = coord_keys
        if xk in ad.obs.columns and yk in ad.obs.columns:
            xy = ad.obs[[xk, yk]].to_numpy()
        elif obsm_key in ad.obsm_keys():
            xy = np.asarray(ad.obsm[obsm_key])[:, :2]
        else:
            raise KeyError(
                f"Could not find coords in ad.obs[{coord_keys}] nor ad.obsm['{obsm_key}']."
            )
        return np.asarray(xy, dtype=np.float64)

    # -------------------------
    # decide order of slices
    # -------------------------
    slice_indices = list(range(len(adata_list)))
    if order_key is not None:
        order_vals = []
        usable = True
        for i, ad in enumerate(adata_list):
            if order_key not in ad.obs.columns:
                usable = False
                break
            v = pd.unique(ad.obs[order_key])
            if len(v) != 1:
                usable = False
                break
            v = v[0]
            if order_map is not None:
                if v not in order_map:
                    usable = False
                    break
                v = order_map[v]
            order_vals.append(v)
        if usable:
            slice_indices = [i for i, _ in sorted(zip(slice_indices, order_vals), key=lambda t: t[1])]

    # -------------------------
    # pass 1: gather per-slice centered coords, widths, and categories
    # -------------------------
    centered = []
    widths = []
    categories_all = []

    for i in slice_indices:
        ad = adata_list[i]
        xy = _get_xy(ad)
        xy = xy - xy.mean(axis=0, keepdims=True)
        xy[:, 1] *= float(y_scale)

        if obs_key not in ad.obs.columns:
            raise KeyError(f"obs_key='{obs_key}' not found in adata.obs for slice index {i}.")

        c = ad.obs[obs_key]
        c = c.astype("category") if not pd.api.types.is_categorical_dtype(c) else c
        categories_all.extend(list(c.cat.categories))

        w = float(np.nanmax(xy[:, 0]) - np.nanmin(xy[:, 0]))
        widths.append(w)

        centered.append((xy, c, ad))

    if x_gap is None:
        med_w = np.median([w for w in widths if np.isfinite(w) and w > 0]) if len(widths) else 1.0
        x_gap = 1.2 * (med_w if np.isfinite(med_w) and med_w > 0 else 1.0)

    # -------------------------
    # global category set (stable by first appearance) as fallback
    # -------------------------
    cat_order_fallback = []
    seen = set()
    for cat in categories_all:
        if cat not in seen:
            seen.add(cat)
            cat_order_fallback.append(cat)

    # -------------------------
    # NEW: alignment to truth for colors/order
    # -------------------------
    # ---- inside your function, replace the alignment block with this ----
    alignment = None
    contingency = None
    if align_colors and (truth_key is not None) and (obs_key == truth_key):
        # No matching needed; keep a stable order and a single palette.
        # Choose ONE palette policy and use it for both aligned/non-aligned.
        cat_order = cat_order_fallback  # or: list(pd.Categorical(...).categories) if you prefer
        cmap = plt.get_cmap(truth_cmap)  # or "tab20" if that’s what you want everywhere
        color_map = {cat: cmap(i % getattr(cmap, "N", 20)) for i, cat in enumerate(cat_order)}
        alignment = {cat: cat for cat in cat_order}
        contingency = None
    elif align_colors and (truth_key is not None):
        for i in slice_indices:
            if truth_key not in adata_list[i].obs.columns:
                raise KeyError(f"truth_key='{truth_key}' not found in adata.obs for slice index {i}.")

        clusters_global = []
        truths_global = []
        contingency = None

        for i in slice_indices:
            ad = adata_list[i]
            tab, c_cats, t_cats = _safe_crosstab(ad, obs_key, truth_key)

            # update global category unions (preserve first-seen order)
            for x in c_cats:
                if x not in clusters_global:
                    clusters_global.append(x)
            for x in t_cats:
                if x not in truths_global:
                    truths_global.append(x)

            if contingency is None:
                contingency = tab.copy()
            else:
                contingency = contingency.add(tab, fill_value=0)

        # reindex to full unions and FORCE numeric fill
        contingency = contingency.reindex(index=clusters_global, columns=truths_global, fill_value=0.0)
        contingency = contingency.fillna(0.0)  # CRITICAL
        contingency = contingency.astype(float)

        M = contingency.to_numpy()
        # sanitize any remaining weirdness
        M = np.nan_to_num(M, nan=0.0, posinf=0.0, neginf=0.0)

        # if everything is zero (e.g., all labels were NaN and dropped), fall back safely
        if M.sum() <= 0:
            # fall back: stable order by appearance (or by cluster size if you prefer)
            cat_order = cat_order_fallback
            cmap = plt.get_cmap("tab20")
            color_map = {cat: cmap(j % cmap.N) for j, cat in enumerate(cat_order)}
            alignment = {cat: None for cat in cat_order}
        else:
            cost = -M
            row_ind, col_ind = linear_sum_assignment(cost)

            clusters = list(contingency.index)
            truths = list(contingency.columns)

            alignment = {cl: None for cl in clusters}
            for r, c in zip(row_ind, col_ind):
                alignment[clusters[r]] = truths[c]

            truth_rank = {t: k for k, t in enumerate(truths)}
            cat_order = sorted(clusters, key=lambda cl: (truth_rank.get(alignment.get(cl), 10**9), str(cl)))

            # colors: cluster inherits matched truth color; unmatched get fallback
            def _make_palette(labels, cmap_name):
                cmap = plt.get_cmap(cmap_name)
                N = getattr(cmap, "N", 12)
                return {lab: cmap(j % N) for j, lab in enumerate(labels)}

            truth_palette = _make_palette(truths, truth_cmap)
            fallback_cmap = plt.get_cmap(unmatched_cmap)
            fallback_N = getattr(fallback_cmap, "N", 20)

            color_map = {}
            k_un = 0
            for cl in cat_order:
                mt = alignment.get(cl, None)
                if (mt is not None) and (mt in truth_palette):
                    color_map[cl] = truth_palette[mt]
                else:
                    color_map[cl] = fallback_cmap(k_un % fallback_N)
                    k_un += 1
    else:
        # fallback: tab20, stable by appearance
        cat_order = cat_order_fallback
        cmap = plt.get_cmap("tab20")
        color_map = {cat: cmap(i % cmap.N) for i, cat in enumerate(cat_order)}

    # -------------------------
    # plot
    # -------------------------
    n = len(centered)
    if figsize is None:
        figsize = (max(8, 3.0 * n), 6)

    fig, ax = plt.subplots(figsize=figsize)

    xmin, xmax = np.inf, -np.inf
    ymin, ymax = np.inf, -np.inf

    x_shift = 0.0
    handles = {}

    for (xy, c, ad), w in zip(centered, widths):
        x = xy[:, 0] + x_shift
        y = xy[:, 1]

        xmin = min(xmin, np.nanmin(x))
        xmax = max(xmax, np.nanmax(x))
        ymin = min(ymin, np.nanmin(y))
        ymax = max(ymax, np.nanmax(y))

        # enforce global order (some cats may not appear in this slice)
        cats = pd.Categorical(c, categories=cat_order)

        for cat in pd.unique(cats):
            mask = (cats == cat)
            if not np.any(mask):
                continue
            sc = ax.scatter(
                x[mask],
                y[mask],
                s=s,
                alpha=alpha,
                c=[color_map[cat]],
                linewidths=0,
                label=str(cat),
            )
            if cat not in handles:
                handles[cat] = sc

        x_shift += (w + x_gap)

    # padding
    pad_frac = 0.02
    xr = xmax - xmin
    yr = ymax - ymin
    xpad = pad_frac * xr if xr > 0 else 1.0
    ypad = pad_frac * yr if yr > 0 else 1.0

    ax.set_xlim(xmin - xpad, xmax + xpad)
    ax.set_ylim(ymin - ypad, ymax + ypad)

    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("")
    ax.set_ylabel("")

    if legend:
        leg_handles = [handles[cat] for cat in cat_order if cat in handles]
        leg_labels = [str(cat) for cat in cat_order if cat in handles]

        n_items = len(leg_labels)
        ncol = min(n_items, 14) if legend_ncol is None else legend_ncol

        fig.legend(
            leg_handles,
            leg_labels,
            loc=legend_loc,
            bbox_to_anchor=legend_bbox,
            ncol=ncol,
            frameon=False,
            fontsize=legend_font,
            handlelength=0.8,
            handletextpad=0.3,
            columnspacing=0.8,
            borderaxespad=0.0,
            markerscale=5,
        )
        fig.subplots_adjust(top=0.88)

    plt.subplots_adjust(top=0.85)
    fig.tight_layout()

    if return_alignment:
        return fig, ax, alignment, contingency
    return fig, ax



def _safe_crosstab(ad, cluster_key, truth_key):
    df = ad.obs[[cluster_key, truth_key]].copy()
    df = df.dropna(subset=[cluster_key, truth_key])  # CRITICAL
    cser = df[cluster_key].astype("category") if not pd.api.types.is_categorical_dtype(df[cluster_key]) else df[cluster_key]
    tser = df[truth_key].astype("category") if not pd.api.types.is_categorical_dtype(df[truth_key]) else df[truth_key]
    tab = pd.crosstab(cser, tser)
    return tab, list(cser.cat.categories), list(tser.cat.categories)