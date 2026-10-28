"""
Core of the LSGL (Learned-Structure Graph Learning) joint gravity/magnetometry
imputer: model, training, prediction on stations and on new points (regular
grid), plus the data helpers it needs.

Self-contained extract of lsgl_joint_imputer.py + utils.py (only the pieces
used by train_and_grid_lsgl.py), so it does not need torch_geometric.
"""
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors

random.seed(123)
np.random.seed(123)
torch.manual_seed(123)
torch.cuda.manual_seed_all(123)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False



# ======================================================================
# from utils.py
# ======================================================================

def normalization(x, y, z):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    z = np.asarray(z, float)

    # Use only finite values for min/max
    x_finite = x[np.isfinite(x)]
    y_finite = y[np.isfinite(y)]
    z_finite = z[np.isfinite(z)]

    if x_finite.size == 0 or y_finite.size == 0 or z_finite.size == 0:
        raise ValueError("Normalization: one of x/y/z has no finite values")

    coordsx_min, coordsx_max = x_finite.min(), x_finite.max()
    coordsy_min, coordsy_max = y_finite.min(), y_finite.max()
    targets_min, targets_max = z_finite.min(), z_finite.max()

    x_norm = (x - coordsx_min) / (coordsx_max - coordsx_min + 1e-10)
    y_norm = (y - coordsy_min) / (coordsy_max - coordsy_min + 1e-10)

    z_norm = z.copy()
    mask = np.isfinite(z_norm)
    z_norm[mask] = 2 * (z_norm[mask] - targets_min) / (targets_max - targets_min + 1e-10) - 1
    # keep NaNs where they were NaN
    return x_norm, y_norm, z_norm, targets_min, targets_max


def unnormalization(z, targets_min, targets_max):

    z_unnorm = z.copy()
    mask = np.isfinite(z_unnorm)
    z_unnorm[mask] = 0.5*(z_unnorm[mask] + 1) * (targets_max - targets_min) + targets_min
    return z_unnorm


def fourier_features_2d(coords_xy, num_freqs=4):
    """
    coords_xy: (N,2) in ORIGINAL or normalized units [0,1].
    num_freqs: number of frequency bands (e.g. 4 -> 4 * 4 = 16 dims)

    Returns: (N, 4 * num_freqs) array of [sin/cos for x and y].
    """
    coords_xy = np.asarray(coords_xy, float)
    x = coords_xy[:, 0]
    y = coords_xy[:, 1]

    # Normalize to [0,1] box for stable frequencies
    x_min, x_max = x.min(), x.max()
    y_min, y_max = y.min(), y.max()
    x_n = (x - x_min) / (x_max - x_min + 1e-12)
    y_n = (y - y_min) / (y_max - y_min + 1e-12)

    feats = []
    for k in range(num_freqs):
        freq = 2.0 ** k  # 1, 2, 4, 8, ...
        feats.append(np.sin(2 * np.pi * freq * x_n))
        feats.append(np.cos(2 * np.pi * freq * x_n))
        feats.append(np.sin(2 * np.pi * freq * y_n))
        feats.append(np.cos(2 * np.pi * freq * y_n))

    # IMPORTANT: stack along axis=1 → shape (N, 4 * num_freqs)
    pe = np.stack(feats, axis=1)
    return pe


def estimate_snap_tol(coords, q=0.05, cap=None):
    """Use a small quantile of nearest-neighbor distances as snap tolerance."""
    nbrs = NearestNeighbors(n_neighbors=2).fit(coords)
    d, _ = nbrs.kneighbors(coords)
    nn = d[:, 1]  # nearest neighbor distance excluding self
    tol = float(np.quantile(nn, q))
    if cap is not None:
        tol = min(tol, cap)
    return max(tol, 1e-6)


def build_features_from_gravity(coords_all, y_grav_all):
    """
    coords_all: (N,2) in original units
    y_grav_all: (N,) gravity values; NaN where unknown (train/test/grid)

    Returns:
      x_cov: (N, 2) = [grav_z_filled, has_grav_mask]
      g_stats: (mu_g, sd_g) used for z-scoring gravity feature
    """
    y = np.asarray(y_grav_all, float)
    has_g = np.isfinite(y)

    if has_g.any():
        mu, sd = y[has_g].mean(), y[has_g].std() if y[has_g].std() > 1e-8 else 1.0
        g_z = (y - mu) / sd
        g_z[~has_g] = 0.0   # fill unknowns with 0 in z-space
        g_stats = (mu, sd)
    else:
        g_z = np.zeros_like(y)
        g_stats = (0.0, 1.0)

    mask_g = has_g.astype(float)  # 1 if gravity is present at the node, else 0
    x_cov = np.column_stack([g_z, mask_g])
    return x_cov, g_stats


def nearest_obs_distance(coords, mask_obs):
    if not np.any(mask_obs):
        # no observations: return zeros
        return np.zeros(coords.shape[0], dtype=float)
    nbrs = NearestNeighbors(n_neighbors=1).fit(coords[mask_obs])
    dist, _ = nbrs.kneighbors(coords)
    return dist[:, 0]


def merge_grav_mag_points(
    coords_grav, values_grav,
    coords_mag,  values_mag,
    snap_tol=25.0,
    reducer="mean",
    return_assignments=False
):
    """
    Merge gravimetry and magnetometry point sets by snapping near-duplicates
    (within `snap_tol`, same units as coords, typically meters) into a single
    node and aggregating values per cluster.

    Parameters
    ----------
    coords_grav : (Ng, 2) array
        East/North (or x/y) for gravimetry points.
    values_grav : (Ng,) array
        Gravity residual/anomaly values (float).
    coords_mag  : (Nm, 2) array
        East/North (or x/y) for magnetometry points.
    values_mag  : (Nm,) array
        Magnetic anomaly values (float).
    snap_tol : float
        Distance tolerance to consider two points "the same" location.
        Use the same linear units as your coordinates (e.g., meters).
        Typical values: 10–50 m depending on survey spacing.
    reducer : {"mean","median"}
        How to aggregate multiple values that fall into the same snapped location.
    return_assignments : bool
        If True, also return arrays mapping each original point to its merged node index.

    Returns
    -------
    coords_union : (N, 2) array
    y_grav_union : (N,) array (NaN where gravity is missing)
    y_mag_union  : (N,) array (NaN where magnetics is missing)
    (optional) assign_grav : (Ng,) array of cluster indices for each grav point
    (optional) assign_mag  : (Nm,) array of cluster indices for each mag point
    """
    coords_grav = np.asarray(coords_grav, float)
    coords_mag  = np.asarray(coords_mag,  float)
    values_grav = np.asarray(values_grav, float)
    values_mag  = np.asarray(values_mag,  float)

    Ng = coords_grav.shape[0]
    Nm = coords_mag.shape[0]

    # 1) Stack all points; keep a source flag
    P = np.vstack([coords_grav, coords_mag])              # (Ng+Nm, 2)
    src = np.concatenate([np.zeros(Ng, dtype=int),        # 0 = grav
                          np.ones(Nm,  dtype=int)])       # 1 = mag

    # 2) Cluster with DBSCAN to snap points within snap_tol
    # metric='euclidean', min_samples=1 ensures every point is in some cluster
    db = DBSCAN(eps=snap_tol, min_samples=1, metric="euclidean")
    labels = db.fit_predict(P)                            # (Ng+Nm,)
    n_clusters = labels.max() + 1

    # 3) For each cluster, compute representative coordinate and aggregated values
    coords_union = np.zeros((n_clusters, 2), dtype=float)
    y_grav_union = np.full(n_clusters, np.nan, dtype=float)
    y_mag_union  = np.full(n_clusters, np.nan, dtype=float)

    # precompute indices per cluster
    clusters = [np.where(labels == k)[0] for k in range(n_clusters)]

    # aggregation function
    if reducer == "mean":
        agg = np.nanmean
    elif reducer == "median":
        agg = np.nanmedian
    else:
        raise ValueError("reducer must be 'mean' or 'median'")

    for k, idx in enumerate(clusters):
        pts = P[idx]                          # (nk, 2)
        coords_union[k] = np.median(pts, axis=0)   # robust representative

        # values from grav points in this cluster
        g_idx = idx[src[idx] == 0]
        if g_idx.size > 0:
            # map to original grav indices (0..Ng-1)
            g_orig = g_idx  # since grav points are first Ng rows in P
            # aggregate those values
            y_grav_union[k] = agg(values_grav[g_orig])

        # values from mag points in this cluster
        m_idx = idx[src[idx] == 1]
        if m_idx.size > 0:
            # map to original mag indices (0..Nm-1)
            m_orig = m_idx - Ng  # offset by Ng because mag points start after grav
            y_mag_union[k] = agg(values_mag[m_orig])

    if return_assignments:
        assign_grav = labels[:Ng]
        assign_mag = labels[Ng:]
        return coords_union, y_grav_union, y_mag_union, assign_grav, assign_mag
    else:
        return coords_union, y_grav_union, y_mag_union




# ======================================================================
# from lsgl_joint_imputer.py
# ======================================================================

def zscore_np(col):
    col = np.asarray(col, float)
    mu = np.nanmean(col)
    sd = np.nanstd(col)
    if (not np.isfinite(sd)) or sd < 1e-8:
        sd = 1.0
    return (col - mu) / sd, (mu, sd)


def huber_loss(u, y, delta=0.5):
    a = (u - y).abs()
    quad = 0.5 * (a**2) / delta
    lin  = a - 0.5 * delta
    return torch.where(a <= delta, quad, lin)


def graph_dirichlet_energy(u, edge_index, edge_weight):
    # Weighted Dirichlet: mean(w_ij (u_i-u_j)^2)
    i, j = edge_index
    return (edge_weight * (u[i] - u[j]).pow(2)).mean()


def graph_biharmonic_energy_norm(u, edge_index, edge_weight, N, eps=1e-12):
    i, j = edge_index
    Lu = torch.zeros((N,), device=u.device, dtype=u.dtype)
    deg = torch.zeros((N,), device=u.device, dtype=u.dtype)
    Lu.index_add_(0, i, edge_weight * (u[i] - u[j]))
    deg.index_add_(0, i, edge_weight)
    Lu = Lu / (deg + eps)
    return (Lu**2).mean()


def edge_alignment_loss(u, v, edge_index, edge_weight=None, eps=1e-8):
    # Encourage shared boundaries (graph cross-gradient)
    i, j = edge_index
    du = u[i] - u[j]
    dv = v[i] - v[j]
    du_n = du / (du.abs() + eps)
    dv_n = dv / (dv.abs() + eps)
    loss = (du_n - dv_n).pow(2)
    if edge_weight is not None:
        loss = loss * edge_weight
    return loss.mean()


def weight_entropy(edge_weight, N, k, eps=1e-12):
    # edge_weight is (N*k,) from row-softmax
    W = edge_weight.view(N, k)
    return -(W * (W + eps).log()).sum(dim=1).mean()


def edge_distance_penalty(coords, edge_index, edge_weight):
    i, j = edge_index
    d2 = (coords[i] - coords[j]).pow(2).sum(dim=1)
    return (edge_weight * d2).mean()


class SharedEncoder(nn.Module):
    def __init__(self, in_dim, hidden=128, z_dim=64, dropout=0.15):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, z_dim),
        )
    def forward(self, x):
        return self.net(x)  # (N,z_dim)


class GraphLearner(nn.Module):
    """
    Learns topology + weights:
      - dense pairwise scores S_ij
      - keep top-k per node
      - row-softmax weights
    Suitable for N ~ up to a few thousands.
    """
    def __init__(self, z_dim, hidden=128):
        super().__init__()
        self.phi = nn.Sequential(
            nn.Linear(3*z_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1)
        )

    def forward(self, z, k=16, tau=1.0):
        N, d = z.shape

        zi = z.unsqueeze(1).expand(N, N, d)
        zj = z.unsqueeze(0).expand(N, N, d)
        feat = torch.cat([zi, zj, (zi - zj).abs()], dim=-1)  # (N,N,3d)

        S = self.phi(feat).squeeze(-1) / tau  # (N,N)
        S.fill_diagonal_(-1e9)

        topk = torch.topk(S, k=k, dim=1)
        dst = topk.indices                          # (N,k)
        src = torch.arange(N, device=z.device).unsqueeze(1).expand(N, k)

        w = torch.softmax(topk.values, dim=1)        # (N,k)

        edge_index = torch.stack([src.reshape(-1), dst.reshape(-1)], dim=0)
        edge_weight = w.reshape(-1)

        return edge_index, edge_weight


class WeightedMP(nn.Module):
    """
    Weighted message passing (no PyG required):
      m_i = sum_j w_ij * h_j
      h_i <- MLP([h_i, m_i])
    """
    def __init__(self, in_dim, hidden=128, num_layers=3, dropout=0.15):
        super().__init__()
        self.num_layers = num_layers
        self.dropout = dropout

        self.blocks = nn.ModuleList()
        self.skips = nn.ModuleList()

        for li in range(num_layers):
            din = (in_dim if li == 0 else hidden)
            self.blocks.append(nn.Sequential(
                nn.Linear(din * 2, hidden),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden, hidden),
                nn.ReLU(),
            ))

            # Skip path: identity if sizes match; else linear projection
            if din == hidden:
                self.skips.append(nn.Identity())
            else:
                self.skips.append(nn.Linear(din, hidden))

    def forward(self, x, edge_index, edge_weight):
        h = x
        src, dst = edge_index  # edges i->j: src=i, dst=j (aggregate dst into src)
        N = h.shape[0]

        for blk, skip in zip(self.blocks, self.skips):
            h_in = h
            m = torch.zeros((N, h.shape[1]), device=h.device, dtype=h.dtype)
            m.index_add_(0, src, edge_weight.unsqueeze(1) * h[dst])

            h_new = blk(torch.cat([h, m], dim=1))
            h = h_new + skip(h_in)
            h = F.dropout(h, p=self.dropout, training=self.training)
        return h


class JointGraphImputer_v1(nn.Module):
    def __init__(self, x_pred_dim, x_graph_dim, enc_hidden=128, z_dim=64, mp_hidden=128, mp_layers=3, dropout=0.15):
        super().__init__()
        self.encoder = SharedEncoder(x_graph_dim, hidden=enc_hidden, z_dim=z_dim, dropout=dropout)
        self.graph_learner = GraphLearner(z_dim=z_dim, hidden=enc_hidden)
        self.mp = WeightedMP(in_dim=x_pred_dim, hidden=mp_hidden, num_layers=mp_layers, dropout=dropout)
        self.head = nn.Sequential(
            nn.Linear(mp_hidden, mp_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(mp_hidden, 2)
        )

    def forward(self, x_pred, x_graph, k_graph=16, tau=1.0):
        z = self.encoder(x_graph)
        edge_index, edge_weight = self.graph_learner(z, k=k_graph, tau=tau)
        h = self.mp(x_pred, edge_index, edge_weight)
        out = self.head(h)
        return out, edge_index, edge_weight

    def forward_with_graph(self, x_pred, edge_index, edge_weight):
        h = self.mp(x_pred, edge_index, edge_weight)
        out = self.head(h)
        return out


def train_imputer_bivariate_lsgl(
    coords_xy,
    y_grav_obs,
    y_mag_obs,
    est_grav=False,
    x_cov=None,
    x_cov_var=None,
    num_freqs=2,
    k_graph=16,
    enc_hidden=128,
    z_dim=64,
    mp_hidden=128,
    mp_layers=3,
    dropout=0.15,
    lr=2e-3,
    weight_decay=4e-4,
    lambda_dirich=1e-4,
    lambda_biharm=1e-4,
    lambda_couple=3e-3,
    lambda_ent=1e-3,
    epochs=3000,
    val_frac=0.03,
    rng_seed=123,
    couple_warmup_frac=0.2,
    outlier_weight_alpha=0.0,
    device="cuda" if torch.cuda.is_available() else "cpu",
    verbose=True,
):
    warmup_graph = int(7 * epochs)
    T_refresh = 5  # recompute graph every 5 epochs
    edge_index_cache = None
    edge_weight_cache = None

    coords_xy = np.asarray(coords_xy, float)
    N = coords_xy.shape[0]
    assert y_grav_obs.shape[0] == N and y_mag_obs.shape[0] == N

    # coords para penalización (normalizadas)
    xy_mu = coords_xy.mean(axis=0, keepdims=True)
    xy_sd = coords_xy.std(axis=0, keepdims=True)
    xy_sd[xy_sd < 1e-8] = 1.0
    coords_norm = (coords_xy - xy_mu) / xy_sd
    t_coords = torch.tensor(coords_norm, dtype=torch.float32, device=device)

    # ---- Features: [coords, x_cov] z-scored column-wise
    feats = coords_xy.copy() if x_cov is None else np.hstack([coords_xy, np.asarray(x_cov, float)])
    X_cols, X_stats = [], []
    for j in range(feats.shape[1]):
        xj, st = zscore_np(feats[:, j])
        X_cols.append(xj[:, None])
        X_stats.append(st)
    X = np.concatenate(X_cols, axis=1)  # (N,F)

    pe_small = fourier_features_2d(coords_xy, num_freqs=num_freqs)  # small PE

    if x_cov_var is None:
        feats_graph = np.hstack([coords_xy, pe_small])  # ONLY geometry
    else:
        feats_graph = np.hstack([coords_xy, pe_small, np.asarray(x_cov_var, float)])

    Xg_cols, Xg_stats = [], []
    for j in range(feats_graph.shape[1]):
        xj, st = zscore_np(feats_graph[:, j])
        Xg_cols.append(xj[:, None])
        Xg_stats.append(st)
    X_graph = np.concatenate(Xg_cols, axis=1)  # (N, F_graph)

    # ---- Targets: z-score separately on observed nodes
    yG_raw = np.asarray(y_grav_obs, float)
    yM_raw = np.asarray(y_mag_obs,  float)
    mask_g = np.isfinite(yG_raw)
    mask_m = np.isfinite(yM_raw)

    mu_g = yG_raw[mask_g].mean() if mask_g.any() else 0.0
    sd_g = yG_raw[mask_g].std()  if mask_g.any() else 1.0
    sd_g = sd_g if sd_g > 1e-8 else 1.0

    mu_m = yM_raw[mask_m].mean() if mask_m.any() else 0.0
    sd_m = yM_raw[mask_m].std()  if mask_m.any() else 1.0
    sd_m = sd_m if sd_m > 1e-8 else 1.0

    yG = np.full(N, np.nan, float)
    yM = np.full(N, np.nan, float)
    if mask_g.any():
        yG[mask_g] = (yG_raw[mask_g] - mu_g) / sd_g
    if mask_m.any():
        yM[mask_m] = (yM_raw[mask_m] - mu_m) / sd_m

    # ---- Internal train/val split (per variable)
    rng = np.random.default_rng(rng_seed)

    def split_mask(mask_obs):
        idx = np.where(mask_obs)[0]
        if idx.size == 0 or val_frac <= 0.0:
            return np.zeros(N, bool), np.zeros(N, bool)
        n_val = int(val_frac * idx.size)
        if n_val <= 0:
            # allow 0 val -> scheduler will use train metric
            m_tr = mask_obs.copy()
            m_val = np.zeros(N, bool)
            return m_tr, m_val
        perm = rng.permutation(idx)
        val_idx = perm[:n_val]
        tr_idx  = perm[n_val:]
        m_tr = np.zeros(N, bool); m_val = np.zeros(N, bool)
        m_tr[tr_idx] = True
        m_val[val_idx] = True
        return m_tr, m_val

    mask_g_tr, mask_g_val = split_mask(mask_g)
    mask_m_tr, mask_m_val = split_mask(mask_m)
    use_val = (mask_g_val.any() or mask_m_val.any())

    # ---- Torch
    x = torch.tensor(X, dtype=torch.float32, device=device)
    x_graph = torch.tensor(X_graph, dtype=torch.float32, device=device)

    t_yG = torch.tensor(yG, dtype=torch.float32, device=device)
    t_yM = torch.tensor(yM, dtype=torch.float32, device=device)

    mg_tr = torch.tensor(mask_g_tr, dtype=torch.bool, device=device)
    mm_tr = torch.tensor(mask_m_tr, dtype=torch.bool, device=device)
    mg_val = torch.tensor(mask_g_val, dtype=torch.bool, device=device)
    mm_val = torch.tensor(mask_m_val, dtype=torch.bool, device=device)

    """model = JointGraphImputer(
        x_in_dim=x.shape[1],
        enc_hidden=enc_hidden,
        z_dim=z_dim,
        mp_hidden=mp_hidden,
        mp_layers=mp_layers,
        dropout=dropout
    ).to(device)"""

    model = JointGraphImputer_v1(
        x_pred_dim=x.shape[1],
        x_graph_dim=x_graph.shape[1],
        enc_hidden=enc_hidden,
        z_dim=z_dim,
        mp_hidden=mp_hidden,
        mp_layers=mp_layers,
        dropout=dropout
    ).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=120, min_lr=1e-5, verbose=verbose)

    best_metric = float("inf")
    best_state = None
    couple_warmup = int(couple_warmup_frac * epochs)

    if est_grav:
        lambda_grav = 1.0
        lambda_magn = 1e-3
    else:
        lambda_grav = 1e-3
        lambda_magn = 1.0

    for ep in range(1, epochs + 1):
        model.train()
        opt.zero_grad()

        t = ep / epochs
        if t < 0.2:
            lambda_dirich = 3e-4  # 3e-4
            lambda_biharm = 1e-5  # 1e-5
            lambda_dist = 1e-1    # 1e-1
        else:
            lambda_dirich = 3e-3  # 3e-3
            lambda_biharm = 1e-4  # 1e-4
            lambda_dist = 9e-2    # 9e-2

        tau = 5.0 if ep < 0.3 * epochs else (2.0 if ep < 0.7 * epochs else 1.0)
        tau = 1

        if ep <= warmup_graph:
            out, edge_index, edge_weight = model(x, x_graph, k_graph=k_graph, tau=tau)
        else:
            if (ep == 1) or (ep % T_refresh == 1) or (edge_index_cache is None):
                z = model.encoder(x_graph)  # (N, z_dim)
                edge_index_cache, edge_weight_cache = model.graph_learner(z, k=k_graph)

                out = model.forward_with_graph(x, edge_index_cache, edge_weight_cache)
                edge_index, edge_weight = edge_index_cache, edge_weight_cache

            else:
                # ---- BETWEEN STEPS: usas el grafo fijo (detach) ----
                out = model.forward_with_graph(
                    x,
                    edge_index_cache,  # edge_index no requiere grad
                    edge_weight_cache.detach()  # IMPORTANT: evita backward por segunda vez
                )
                edge_index, edge_weight = edge_index_cache, edge_weight_cache.detach()

        predG, predM = out[:, 0], out[:, 1]

        #out, edge_index, edge_weight = model(x, k_graph=k_graph)
        #out, edge_index, edge_weight = model(x, x_graph, k_graph=k_graph, tau=1.0)
        #predG, predM = out[:, 0], out[:, 1]

        # data fit (Huber) on TRAIN observed, optionally up-weighted by target
        # magnitude (z-scored) so extreme/outlier stations get a stronger
        # gradient signal than Huber's own delta-clipping would otherwise
        # allow -- targets this specific peak-underfitting failure mode
        # without touching the smoothness regularization (lambda_dirich/
        # lambda_biharm), which is what keeps the rest of the surface smooth.
        if mg_tr.any():
            elem_g = huber_loss(predG[mg_tr], t_yG[mg_tr], delta=0.3)
            w_g = 1.0 + outlier_weight_alpha * t_yG[mg_tr].abs()
            loss_g = (elem_g * w_g).sum() / w_g.sum()
        else:
            loss_g = predG.new_tensor(0.0)

        if mm_tr.any():
            elem_m = huber_loss(predM[mm_tr], t_yM[mm_tr], delta=0.5)
            w_m = 1.0 + outlier_weight_alpha * t_yM[mm_tr].abs()
            loss_m = (elem_m * w_m).sum() / w_m.sum()
        else:
            loss_m = predM.new_tensor(0.0)

        # smoothness on learned graph
        Eg = graph_dirichlet_energy(predG, edge_index, edge_weight)
        Em = graph_dirichlet_energy(predM, edge_index, edge_weight)

        Bg = graph_biharmonic_energy_norm(predG, edge_index, edge_weight, N=N)
        Bm = graph_biharmonic_energy_norm(predM, edge_index, edge_weight, N=N)

        loss = lambda_grav*loss_g + lambda_magn*loss_m
        #loss = loss + lambda_dirich * (Eg + Em)
        #loss = loss + lambda_biharm * (Bg + Bm)

        if est_grav:
            loss = loss + lambda_dirich * (Eg)
            loss = loss + lambda_biharm * (Bg)
            # loss = loss + lambda_dirich * (Eg + Em)
            # loss = loss + lambda_biharm * (Bg + Bm)
        else:
            loss = loss + lambda_dirich * (Em)
            loss = loss + lambda_biharm * (Bm)

        #loss += (0.2 * lambda_dirich) * Eg
        #loss += (0.2 * lambda_biharm) * Bg

        # coupling (shared boundaries) after warmup
        if lambda_couple > 0.0 and ep > couple_warmup:
            loss = loss + lambda_couple * edge_alignment_loss(predG, predM, edge_index, edge_weight=edge_weight)

        # encourage confident neighbor selection (lower entropy)
        H = weight_entropy(edge_weight, N=N, k=k_graph)
        H_target = 0.6 * np.log(k_graph)   # 0.6
        loss = loss + lambda_ent * (H - H_target).pow(2)
        # loss = loss + lambda_ent * weight_entropy(edge_weight, N=N, k=k_graph)

        #lambda_dist = 1e-2
        Ldist = edge_distance_penalty(t_coords, edge_index, edge_weight)
        loss = loss + lambda_dist * Ldist

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if edge_weight_cache is not None:
            edge_weight_cache = edge_weight_cache.detach()

        # metric: val MSE if available, else train loss
        model.eval()
        with torch.no_grad():
            if edge_index_cache is None:
                z = model.encoder(x_graph)
                edge_index_cache, edge_weight_cache = model.graph_learner(z, k=k_graph)

            # outv, _, _ = model(x, k_graph=k_graph)
            # outv, _, _ = model(x, x_graph, k_graph=k_graph)
            outv = model.forward_with_graph(x, edge_index_cache, edge_weight_cache)
            g_v, m_v = outv[:, 0], outv[:, 1]

            if use_val:
                val_g = F.mse_loss(g_v[mg_val], t_yG[mg_val]) if mg_val.any() else g_v.new_tensor(0.0)
                val_m = F.mse_loss(m_v[mm_val], t_yM[mm_val]) if mm_val.any() else m_v.new_tensor(0.0)
                # metric = val_g + val_m

                if est_grav:
                    metric = val_g
                else:
                    metric = val_m
            else:
                metric = loss.detach()

        sched.step(metric)

        if metric.item() < best_metric:
            best_metric = metric.item()
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        if verbose and (ep == 1 or ep % 50 == 0):
            tag = "val" if use_val else "train"
            print(f"[{ep:04d}] loss={loss.item():.4f} | g={loss_g.item():.4f} m={loss_m.item():.4f} | {tag}={metric.item():.4f}")

        if ep % 50 == 0:
            print("data", float((lambda_grav * loss_g + loss_m).item()),
                  "dist", float((lambda_dist * Ldist).item()),
                  "dir", float((lambda_dirich * Em).item()),
                  "bih", float((lambda_biharm * Bm).item()))

    if best_state is not None:
        model.load_state_dict(best_state)

    stats = ((mu_g, sd_g), (mu_m, sd_m))
    return model, stats, X_stats, Xg_stats, (mask_g, mask_m), X, X_graph


@torch.no_grad()
def predict_lsgl_mc_dual(
    model,
    coords_xy,
    x_cov,
    x_cov_var,
    X_stats_pred,   # stats for x_pred (coords + x_cov)
    X_stats_graph,  # stats for x_graph (coords + pe_small)
    stats_targets,  # ((mu_g, sd_g), (mu_m, sd_m))
    k_graph=16,
    n_mc=64,
    pe_num_freqs_graph=1,
    tau=1.0,
    device="cuda",
    refresh_graph_each_mc=False,   # <- opcional
):
    """
    Dual-input inference with MC dropout.
    Returns:
      (g_mean,g_std), (m_mean,m_std) in ORIGINAL units
    """
    model = model.to(device)
    model.train()  # MC dropout ON

    coords_xy = np.asarray(coords_xy, float)

    # ---------- Build x_pred = [coords, x_cov] ----------
    feats_pred = coords_xy.copy() if x_cov is None else np.hstack([coords_xy, np.asarray(x_cov, float)])

    Xp_cols = []
    for j in range(feats_pred.shape[1]):
        mu, sd = X_stats_pred[j]
        sd = sd if sd > 1e-8 else 1.0
        Xp_cols.append(((feats_pred[:, j] - mu) / sd)[:, None])
    X_pred = np.concatenate(Xp_cols, axis=1)

    # ---------- Build x_graph = [coords, small PE] ----------
    pe_small = fourier_features_2d(coords_xy, num_freqs=pe_num_freqs_graph)

    if x_cov_var is None:
        feats_graph = np.hstack([coords_xy, pe_small])
    else:
        feats_graph = np.hstack([coords_xy, pe_small, x_cov_var])

    Xg_cols = []
    for j in range(feats_graph.shape[1]):
        mu, sd = X_stats_graph[j]
        sd = sd if sd > 1e-8 else 1.0
        Xg_cols.append(((feats_graph[:, j] - mu) / sd)[:, None])
    X_graph = np.concatenate(Xg_cols, axis=1)

    # ---------- Torch ----------
    x_pred = torch.tensor(X_pred,  dtype=torch.float32, device=device)
    x_graph = torch.tensor(X_graph, dtype=torch.float32, device=device)

    # ---------- Build graph ONCE (recommended) ----------
    if not refresh_graph_each_mc:
        z = model.encoder(x_graph)
        edge_index, edge_weight = model.graph_learner(z, k=k_graph, tau=tau)  # if tau supported
        # safety: avoid any accidental autograd baggage (though we're in no_grad)
        edge_weight = edge_weight.detach()

    # ---------- MC predictions ----------
    preds = []
    for _ in range(n_mc):
        if refresh_graph_each_mc:
            z = model.encoder(x_graph)
            edge_index, edge_weight = model.graph_learner(z, k=k_graph, tau=tau)
            edge_weight = edge_weight.detach()

        out = model.forward_with_graph(x_pred, edge_index, edge_weight)
        preds.append(out.detach().cpu().numpy())

    P = np.stack(preds, axis=0)  # (mc,N,2)
    mean_z = P.mean(axis=0)
    std_z  = P.std(axis=0, ddof=1)

    (mu_g, sd_g), (mu_m, sd_m) = stats_targets
    g_mean = mean_z[:, 0] * sd_g + mu_g
    g_std  = std_z[:, 0]  * sd_g
    m_mean = mean_z[:, 1] * sd_m + mu_m
    m_std  = std_z[:, 1]  * sd_m
    return (g_mean, g_std), (m_mean, m_std)


def _build_pred_and_graph_features(coords, x_cov, x_cov_var, X_stats_pred, X_stats_graph,
                                    pe_num_freqs_graph, device):
    """Same feature construction as predict_lsgl_mc_dual, factored out so it
    can be called separately for the station set and for a grid chunk."""
    feats_pred = coords.copy() if x_cov is None else np.hstack([coords, np.asarray(x_cov, float)])
    cols = []
    for j in range(feats_pred.shape[1]):
        mu, sd = X_stats_pred[j]
        sd = sd if sd > 1e-8 else 1.0
        cols.append(((feats_pred[:, j] - mu) / sd)[:, None])
    X_pred = np.concatenate(cols, axis=1)

    pe = fourier_features_2d(coords, num_freqs=pe_num_freqs_graph)
    feats_graph = np.hstack([coords, pe]) if x_cov_var is None else np.hstack([coords, pe, np.asarray(x_cov_var, float)])
    cols_g = []
    for j in range(feats_graph.shape[1]):
        mu, sd = X_stats_graph[j]
        sd = sd if sd > 1e-8 else 1.0
        cols_g.append(((feats_graph[:, j] - mu) / sd)[:, None])
    X_graph = np.concatenate(cols_g, axis=1)

    x_pred = torch.tensor(X_pred, dtype=torch.float32, device=device)
    x_graph = torch.tensor(X_graph, dtype=torch.float32, device=device)
    return x_pred, x_graph


@torch.no_grad()
def predict_on_grid_attached(
    model,
    coords_stations, x_cov_stations, x_cov_var_stations,
    coords_grid, x_cov_grid, x_cov_var_grid,
    X_stats_pred, X_stats_graph, stats_targets,
    k_graph=16, n_mc=64, pe_num_freqs_graph=1, device="cuda",
):
    """
    Predicts on NEW coordinates (e.g. a regular grid) by attaching them to
    the graph already learned over the station nodes, instead of rebuilding
    a dense graph over stations+grid together (which is what
    predict_lsgl_mc_dual does, and is O((N_stations+N_grid)^2) --
    GraphLearner's own docstring caps that at "a few thousand" nodes).

    Cost here: O(N_stations^2) once (cheap, same as training) +
    O(N_grid * N_stations) to attach the grid (linear in grid size, since
    each grid node only scores against the stations, never against other
    grid nodes). Safe for grids much larger than GraphLearner's own limit.

    Returns (g_mean, g_std), (m_mean, m_std) for the GRID points only, in
    original units.
    """
    model = model.to(device)

    # ---- stations: rebuild the SAME graph learned in training (fixed edges) ----
    x_pred_st, x_graph_st = _build_pred_and_graph_features(
        coords_stations, x_cov_stations, x_cov_var_stations,
        X_stats_pred, X_stats_graph, pe_num_freqs_graph, device,
    )
    z_st = model.encoder(x_graph_st)
    edge_index_st, edge_weight_st = model.graph_learner(z_st, k=k_graph)
    N_st = x_pred_st.shape[0]

    # ---- grid: attach each grid node to its top-k STATIONS only (rectangular, not N^2) ----
    x_pred_gr, x_graph_gr = _build_pred_and_graph_features(
        coords_grid, x_cov_grid, x_cov_var_grid,
        X_stats_pred, X_stats_graph, pe_num_freqs_graph, device,
    )
    z_gr = model.encoder(x_graph_gr)
    N_gr = x_pred_gr.shape[0]

    # reuse the SAME trained scoring MLP, but only grid-vs-station pairs
    zi = z_gr.unsqueeze(1).expand(N_gr, N_st, z_gr.shape[1])
    zj = z_st.unsqueeze(0).expand(N_gr, N_st, z_st.shape[1])
    feat = torch.cat([zi, zj, (zi - zj).abs()], dim=-1)
    S = model.graph_learner.phi(feat).squeeze(-1)  # (N_grid, N_st)

    # A grid point that coincides (near-exactly) with a station must not be
    # allowed to pick ITSELF as a neighbor -- GraphLearner excludes the
    # diagonal the same way (S.fill_diagonal_(-1e9)) when stations score
    # against each other; this is the rectangular equivalent, keyed off the
    # learned embedding (identical coords + covariates -> identical z).
    same_point = torch.cdist(z_gr, z_st, p=2) < 1e-6
    S = S.masked_fill(same_point, -1e9)

    topk = torch.topk(S, k=k_graph, dim=1)
    w_grid = torch.softmax(topk.values, dim=1)     # (N_grid, k) -- grid node -> its k station neighbors

    # grid nodes are appended AFTER stations in the combined node set
    src_gr = torch.arange(N_gr, device=device).unsqueeze(1).expand(N_gr, k_graph).reshape(-1) + N_st
    dst_gr = topk.indices.reshape(-1)
    w_gr = w_grid.reshape(-1)

    src_st, dst_st = edge_index_st
    edge_index_all = torch.cat([torch.stack([src_st, dst_st], 0), torch.stack([src_gr, dst_gr], 0)], dim=1)
    edge_weight_all = torch.cat([edge_weight_st, w_gr], dim=0)
    x_pred_all = torch.cat([x_pred_st, x_pred_gr], dim=0)

    # ---- MC-dropout forward through the (frozen) message passing + head ----
    model.train()  # keep dropout ON for MC
    preds = []
    for _ in range(n_mc):
        h = model.mp(x_pred_all, edge_index_all, edge_weight_all)
        out = model.head(h)
        preds.append(out.detach().cpu().numpy())
    P = np.stack(preds, axis=0)  # (mc, N_st+N_grid, 2)
    mean_z = P.mean(axis=0)[N_st:]  # keep grid rows only
    std_z = P.std(axis=0, ddof=1)[N_st:]

    (mu_g, sd_g), (mu_m, sd_m) = stats_targets
    g_mean = mean_z[:, 0] * sd_g + mu_g
    g_std  = std_z[:, 0]  * sd_g
    m_mean = mean_z[:, 1] * sd_m + mu_m
    m_std  = std_z[:, 1]  * sd_m
    return (g_mean, g_std), (m_mean, m_std)


def load_lsgl_checkpoint(path, device="cuda" if torch.cuda.is_available() else "cpu"):
    """
    Reloads a checkpoint saved by the training block below (the
    `./data_results/lsgl_joint_model_*.pt` files) and rebuilds a ready-to-use
    JointGraphImputer_v1, already in eval mode on `device`.

    Returns a dict with everything predict_on_grid_attached needs:
      model, X_stats, Xg_stats, stats_targets, coords_union, x_cov,
      x_cov_var, k_graph, num_freqs -- plus the run metadata
      (label_var, data_name, sampling, test_size, rmse_grav, rmse_mag)
      that was stored alongside it.

    Example:
        ckpt = load_lsgl_checkpoint(
            "./data_results/lsgl_joint_model_gravimetry_real_2_random_test=0.2.pt"
        )
        (g_mean, g_std), (m_mean, m_std) = predict_on_grid_attached(
            model=ckpt["model"],
            coords_stations=ckpt["coords_union"], x_cov_stations=ckpt["x_cov"],
            x_cov_var_stations=ckpt["x_cov_var"],
            coords_grid=grid_coords, x_cov_grid=x_cov_grid, x_cov_var_grid=x_cov_var_grid,
            X_stats_pred=ckpt["X_stats"], X_stats_graph=ckpt["Xg_stats"],
            stats_targets=ckpt["stats_targets"],
            k_graph=ckpt["k_graph"], pe_num_freqs_graph=ckpt["num_freqs"],
            device=device,
        )
    """
    ckpt = torch.load(path, map_location=device, weights_only=False)

    model = JointGraphImputer_v1(**ckpt["model_init_kwargs"]).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    return {
        "model": model,
        "X_stats": ckpt["X_stats"],
        "Xg_stats": ckpt["Xg_stats"],
        "stats_targets": ckpt["stats_targets"],
        "coords_union": ckpt["coords_union"],
        "x_cov": ckpt["x_cov"],
        "x_cov_var": ckpt["x_cov_var"],
        "k_graph": ckpt["k_graph"],
        "num_freqs": ckpt["num_freqs"],
        "label_var": ckpt["label_var"],
        "data_name": ckpt["data_name"],
        "sampling": ckpt["sampling"],
        "test_size": ckpt["test_size"],
        "rmse_grav": ckpt["rmse_grav"],
        "rmse_mag": ckpt["rmse_mag"],
        # Absent in checkpoints saved before this field was added -- falls
        # back to None, which build_grid_covariates() below refuses to run
        # on (it needs them to build faithful features for new points).
        "y_grav_all": ckpt.get("y_grav_all"),
        "y_mag_all": ckpt.get("y_mag_all"),
        "grav_inFeat": ckpt.get("grav_inFeat"),
        "mag_inFeat": ckpt.get("mag_inFeat"),
    }


def build_grid_covariates(ckpt, coords_new):
    """
    Builds x_cov / x_cov_var for NEW coordinates (e.g. a regular grid) that
    exactly match how x_cov / x_cov_var were built for the stations during
    training (see the "both" branch in __main__ below) -- same column
    order, same "value + has-observation mask" convention (0/0 for the new
    points, since they have no real measurement), and distances computed
    against the SAME reference stations used in training.

    Requires a checkpoint saved with y_grav_all / y_mag_all (see
    load_lsgl_checkpoint) -- older checkpoints don't have them, in which
    case this raises instead of silently approximating.
    """
    if ckpt.get("y_grav_all") is None or ckpt.get("y_mag_all") is None:
        raise ValueError(
            "This checkpoint doesn't have y_grav_all/y_mag_all saved "
            "(it was created before build_grid_covariates existed). "
            "Re-run training to get a checkpoint with these fields."
        )

    coords_stations = ckpt["coords_union"]
    y_grav_all = ckpt["y_grav_all"]
    y_mag_all = ckpt["y_mag_all"]
    grav_inFeat = ckpt["grav_inFeat"]
    mag_inFeat = ckpt["mag_inFeat"]
    num_freqs = ckpt["num_freqs"]
    coords_new = np.asarray(coords_new, float)
    N_new = coords_new.shape[0]

    # ---- same "value + has-observation mask" pair as build_features_from_gravity,
    # but for points with NO real measurement: value=0, mask=0 ----
    if grav_inFeat and mag_inFeat:
        value_mask_new = np.zeros((N_new, 4))  # [grav_z=0, has_grav=0, mag_z=0, has_mag=0]
    elif grav_inFeat or mag_inFeat:
        value_mask_new = np.zeros((N_new, 2))  # [value_z=0, has_value=0]
    else:
        value_mask_new = np.zeros((N_new, 0))

    # ---- d_g / d_m: distance from each new point to the nearest STATION
    # that actually had a grav/mag observation (same reference set used to
    # build d_g/d_m for the stations themselves) ----
    mask_g_obs = np.isfinite(y_grav_all)
    mask_m_obs = np.isfinite(y_mag_all)

    def dist_to_nearest(ref_coords, ref_mask, query_coords):
        if not np.any(ref_mask):
            return np.zeros(query_coords.shape[0], dtype=float)
        nbrs = NearestNeighbors(n_neighbors=1).fit(ref_coords[ref_mask])
        return nbrs.kneighbors(query_coords)[0][:, 0]

    d_g = dist_to_nearest(coords_stations, mask_g_obs, coords_new)
    d_m = dist_to_nearest(coords_stations, mask_m_obs, coords_new)

    pe = fourier_features_2d(coords_new, num_freqs=num_freqs)

    x_cov_new = np.hstack([value_mask_new, d_g[:, None], d_m[:, None], pe])

    # x_cov_var is always built from y_mag_all alone (see __main__ below):
    # [value_z] only, no mask column -- 0 for new points (no measurement).
    x_cov_var_new = np.zeros((N_new, 1))

    return x_cov_new, x_cov_var_new

