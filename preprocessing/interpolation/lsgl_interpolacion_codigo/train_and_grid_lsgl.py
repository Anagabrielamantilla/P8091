"""
Full pipeline in one run: train the LSGL graph model (lsgl_core.py)
on the Cerro Machin stations, hold out a small test fraction to report RMSE,
and predict over a regular grid.

Steps:
  1. Load gravity + magnetometry stations, normalize each to [-1, 1].
  2. Random train/test split on the TARGET variable (TEST_FRAC); the other
     variable is used in full as auxiliary input (same as the original lsgl_joint_imputer.py's
     "both" branch).
  3. Merge grav/mag stations into graph nodes, build covariates, train.
  4. RMSE on the held-out test stations (mGal / nT).
  5. Grid prediction with predict_on_grid_attached (graph model everywhere),
     then residual correction: station residuals (real - model) are
     interpolated linearly and added back, so the final map honors every
     station and keeps the peaks.
  6. Saves checkpoint (.pt, loadable with load_lsgl_checkpoint), grid (.npz +
     .csv) and figures (.png/.svg).

Run:  python train_and_grid_lsgl.py                          (defaults below)
      python train_and_grid_lsgl.py --target magnetometry
      python train_and_grid_lsgl.py --target gravimetry --test-frac 0.1 --seed 7 --nx 200 --ny 200
      python train_and_grid_lsgl.py --data-dir /ruta/a/los/datos --out-dir /ruta/de/salida
"""
import argparse
import os

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from scipy.spatial import Delaunay

from lsgl_core import (
    normalization, unnormalization, estimate_snap_tol, merge_grav_mag_points, build_features_from_gravity, nearest_obs_distance,
    fourier_features_2d, train_imputer_bivariate_lsgl, predict_lsgl_mc_dual,
    load_lsgl_checkpoint, build_grid_covariates, predict_on_grid_attached,
)

# ============================== configuration ==============================
TARGET = "gravimetry"        # "gravimetry" | "magnetometry"
TEST_FRAC = 0.05             # fraction of TARGET stations held out for RMSE
SEED = 123
NX, NY = 100, 100            # grid resolution
EPOCHS = 5000
K_GRAPH = 14
NUM_FREQS = 3
N_MC = 300                   # MC-dropout samples at inference
RESIDUALS_FROM = "all"       # "all" (train+test stations) | "train" (keeps test stations unseen by the final map)

DATA_DIR = "./data"         # folder with GRAV_FILE and MAG_FILE
GRAV_FILE = "Estaciones_AGCM_SGC_Gravimetria.txt"
MAG_FILE = "Estaciones_AGCM_SGC_Magnetometria.txt"
OUT_DIR = "./resultados"     # grids, checkpoint and figures go here
# ===========================================================================

# command-line overrides of the defaults above
parser = argparse.ArgumentParser(description="Train LSGL + predict a regular grid")
parser.add_argument("--target", choices=["gravimetry", "magnetometry"], default=TARGET)
parser.add_argument("--test-frac", type=float, default=TEST_FRAC)
parser.add_argument("--seed", type=int, default=SEED)
parser.add_argument("--nx", type=int, default=NX)
parser.add_argument("--ny", type=int, default=NY)
parser.add_argument("--epochs", type=int, default=EPOCHS)
parser.add_argument("--n-mc", type=int, default=N_MC)
parser.add_argument("--residuals-from", choices=["all", "train"], default=RESIDUALS_FROM)
parser.add_argument("--data-dir", default=DATA_DIR)
parser.add_argument("--out-dir", default=OUT_DIR)
args = parser.parse_args()
DATA_DIR, OUT_DATA = args.data_dir, args.out_dir
OUT_FIG = OUT_DATA
for fname in (GRAV_FILE, MAG_FILE):
    if not os.path.isfile(os.path.join(DATA_DIR, fname)):
        raise SystemExit(f"No se encontró {os.path.join(DATA_DIR, fname)}. "
                         f"Coloque los archivos de estaciones en esa carpeta o use --data-dir.")
TARGET, TEST_FRAC, SEED = args.target, args.test_frac, args.seed
NX, NY, EPOCHS, N_MC, RESIDUALS_FROM = args.nx, args.ny, args.epochs, args.n_mc, args.residuals_from
print(f"Configuración: target={TARGET} test_frac={TEST_FRAC} seed={SEED} malla={NX}x{NY} "
      f"epochs={EPOCHS} n_mc={N_MC} residuos={RESIDUALS_FROM}")

UNITS = {"gravimetry": "mGal", "magnetometry": "nT"}
CBAR_LABEL = {"gravimetry": "Anomalía Gravimétrica (mGal)", "magnetometry": "Anomalía Magnética (nT)"}

device = "cuda" if torch.cuda.is_available() else "cpu"
est_grav = TARGET == "gravimetry"
units = UNITS[TARGET]
tag = f"{TARGET}_test={TEST_FRAC}_seed={SEED}"
os.makedirs(OUT_DATA, exist_ok=True)
os.makedirs(OUT_FIG, exist_ok=True)

# ---- 1. data ----
df_grav = pd.read_csv(os.path.join(DATA_DIR, GRAV_FILE), sep=r"\s+", header=None, names=["X", "Y", "Z", "ARB"])
df_mag = pd.read_csv(os.path.join(DATA_DIR, MAG_FILE), sep=r"\s+", header=None, names=["X", "Y", "Z", "AMCT"])

coords_grav = df_grav[["X", "Y"]].values.astype(float)
coords_mag = df_mag[["X", "Y"]].values.astype(float)
z_grav_raw = df_grav["ARB"].values.astype(float)
z_mag_raw = df_mag["AMCT"].values.astype(float)

_, _, z_grav, z_grav_min, z_grav_max = normalization(coords_grav[:, 0], coords_grav[:, 1], z_grav_raw)
_, _, z_mag, z_mag_min, z_mag_max = normalization(coords_mag[:, 0], coords_mag[:, 1], z_mag_raw)

# ---- 2. train/test split on the target variable only ----
# (simple seeded random split of the target stations)
n_target = len(z_grav) if est_grav else len(z_mag)
rng = np.random.default_rng(SEED)
idx_test = np.sort(rng.choice(n_target, size=max(1, int(round(TEST_FRAC * n_target))), replace=False))
mask_train_target = np.ones(n_target, dtype=bool)
mask_train_target[idx_test] = False
mask_train_grav = mask_train_target if est_grav else np.ones(len(z_grav), dtype=bool)
mask_train_mag = np.ones(len(z_mag), dtype=bool) if est_grav else mask_train_target
print(f"{TARGET}: {n_target} estaciones -> train={mask_train_target.sum()}  test={len(idx_test)}")

# ---- 3. graph nodes + covariates (same construction as the original training script) ----
snap_tol = min(estimate_snap_tol(coords_grav, q=0.05, cap=50.0), estimate_snap_tol(coords_mag, q=0.05, cap=50.0))
coords_union, _, _, assign_grav, assign_mag = merge_grav_mag_points(
    coords_grav, z_grav, coords_mag, z_mag, snap_tol=snap_tol, reducer="mean", return_assignments=True,
)
N_nodes = coords_union.shape[0]


def aggregate(assign, z, mask_train):
    """Per-node mean of the TRAIN observations that snapped to that node (NaN if none)."""
    y = np.full(N_nodes, np.nan)
    for k in range(N_nodes):
        idx = np.where((assign == k) & mask_train)[0]
        if idx.size:
            y[k] = z[idx].mean()
    return y


y_grav_all = aggregate(assign_grav, z_grav, mask_train_grav)
y_mag_all = aggregate(assign_mag, z_mag, mask_train_mag)

# the auxiliary variable goes in as a covariate (value + has-observation mask)
grav_inFeat, mag_inFeat = (False, True) if est_grav else (True, False)
y_aux_all = y_mag_all if est_grav else y_grav_all
x_cov, aux_stats = build_features_from_gravity(coords_union, y_aux_all)
# graph-learner covariate: the AUXILIARY variable's value (never the target's
# own value, which would leak it into the graph for training nodes)
x_cov_var = x_cov[:, :1].copy()

d_g = nearest_obs_distance(coords_union, np.isfinite(y_grav_all))
d_m = nearest_obs_distance(coords_union, np.isfinite(y_mag_all))
pe = fourier_features_2d(coords_union, num_freqs=NUM_FREQS)
x_cov = np.hstack([x_cov, d_g[:, None], d_m[:, None], pe])

# ---- train ----
model_kwargs = dict(enc_hidden=128, z_dim=(30 if est_grav else 64), mp_hidden=128, mp_layers=2, dropout=0.15)
torch.manual_seed(SEED)
model, stats, X_stats, Xg_stats, _, X, X_graph = train_imputer_bivariate_lsgl(
    coords_xy=coords_union, y_grav_obs=y_grav_all, y_mag_obs=y_mag_all, est_grav=est_grav,
    x_cov=x_cov, x_cov_var=x_cov_var, num_freqs=NUM_FREQS, k_graph=K_GRAPH, **model_kwargs,
    lr=4e-3, weight_decay=4e-4, lambda_dirich=3e-4, lambda_biharm=1e-5, lambda_couple=3e-12,
    lambda_ent=1e-4, epochs=EPOCHS, val_frac=0.03, couple_warmup_frac=0.3, rng_seed=SEED,
    device=device, verbose=True,
)

# ---- 4. RMSE on held-out stations ----
(g_mean, _), (m_mean, _) = predict_lsgl_mc_dual(
    model=model, coords_xy=coords_union, x_cov=x_cov, x_cov_var=x_cov_var,
    X_stats_pred=X_stats, X_stats_graph=Xg_stats, stats_targets=stats,
    k_graph=K_GRAPH, n_mc=N_MC, pe_num_freqs_graph=NUM_FREQS, device=device,
)
if est_grav:
    pred_st = unnormalization(g_mean[assign_grav], z_grav_min, z_grav_max)
    z_st_raw, coords_st, z_min, z_max = z_grav_raw, coords_grav, z_grav_min, z_grav_max
else:
    pred_st = unnormalization(m_mean[assign_mag], z_mag_min, z_mag_max)
    z_st_raw, coords_st, z_min, z_max = z_mag_raw, coords_mag, z_mag_min, z_mag_max

err_test = pred_st[idx_test] - z_st_raw[idx_test]
err_train = pred_st[mask_train_target] - z_st_raw[mask_train_target]
rmse_test = float(np.sqrt(np.mean(err_test ** 2)))
rmse_train = float(np.sqrt(np.mean(err_train ** 2)))
print(pd.DataFrame({"idx": idx_test, "X": coords_st[idx_test, 0], "Y": coords_st[idx_test, 1],
                    "real": z_st_raw[idx_test], "pred": pred_st[idx_test], "error": err_test}).round(3).to_string(index=False))
print(f"RMSE test  ({len(idx_test)} estaciones): {rmse_test:.3f} {units}")
print(f"RMSE train ({mask_train_target.sum()} estaciones): {rmse_train:.3f} {units}")

# ---- checkpoint (loadable with load_lsgl_checkpoint) ----
ckpt_path = f"{OUT_DATA}/lsgl_pipeline_model_{tag}.pt"
torch.save({
    "model_state_dict": model.state_dict(),
    "model_init_kwargs": dict(x_pred_dim=X.shape[1], x_graph_dim=X_graph.shape[1], **model_kwargs),
    "X_stats": X_stats, "Xg_stats": Xg_stats, "stats_targets": stats,
    "coords_union": coords_union, "x_cov": x_cov, "x_cov_var": x_cov_var,
    "k_graph": K_GRAPH, "num_freqs": NUM_FREQS,
    "label_var": TARGET, "data_name": "real_2", "sampling": "random", "test_size": TEST_FRAC,
    "rmse_grav": rmse_test if est_grav else None, "rmse_mag": None if est_grav else rmse_test,
    "y_grav_all": y_grav_all, "y_mag_all": y_mag_all,
    "grav_inFeat": grav_inFeat, "mag_inFeat": mag_inFeat,
    "idx_test": idx_test,
}, ckpt_path)
print(f"Checkpoint: {ckpt_path}")

# ---- 5. grid prediction (graph model attached to the learned station graph) ----
ckpt = load_lsgl_checkpoint(ckpt_path, device=device)


# If the auxiliary variable is observed at EVERY training node (e.g. target =
# magnetometry: all nodes carry gravity), the model has never seen a node
# without it (has_aux=1 and d_aux=0 everywhere), so feeding new points
# has_aux=0 (build_grid_covariates' convention) is out of distribution and
# the predictions blow up. In that case the auxiliary value at new points is
# interpolated from the stations instead, with has_aux=1 and d_aux=0 -- the
# same feature distribution the model was trained on.
aux_full_coverage = bool(np.isfinite(y_aux_all).all())
print(f"Variable auxiliar en todos los nodos: {aux_full_coverage} -> covariables en malla: "
      f"{'auxiliar interpolada' if aux_full_coverage else 'build_grid_covariates'}")


def covariates_interp_aux(coords_new):
    obs = np.isfinite(y_aux_all)
    v = griddata(coords_union[obs], y_aux_all[obs], coords_new, method="linear")
    v_near = griddata(coords_union[obs], y_aux_all[obs], coords_new, method="nearest")
    v = np.where(np.isfinite(v), v, v_near)
    mu, sd = aux_stats
    aux_z = (v - mu) / sd
    d_target = nearest_obs_distance(
        np.vstack([coords_union, coords_new]),
        np.r_[np.isfinite(y_grav_all if est_grav else y_mag_all), np.zeros(len(coords_new), bool)],
    )[N_nodes:]
    d_aux = np.zeros(len(coords_new))
    d_g, d_m = (d_target, d_aux) if est_grav else (d_aux, d_target)
    pe_new = fourier_features_2d(coords_new, num_freqs=NUM_FREQS)
    x_cov_new = np.column_stack([aux_z, np.ones(len(coords_new)), d_g, d_m, pe_new])
    return x_cov_new, aux_z[:, None]


def predict_points(coords_new):
    # fourier_features_2d normalizes by the min/max of whatever set it gets, so
    # pad the query with the station-graph bounding-box corners: the positional
    # encoding of new points then uses the SAME box as in training. Grid nodes
    # only attach to stations (never to each other), so the padding rows don't
    # affect the real ones and are dropped at the end.
    lo, hi = coords_union.min(axis=0), coords_union.max(axis=0)
    coords_pad = np.vstack([coords_new, lo, hi])
    if aux_full_coverage:
        x_cov_new, x_cov_var_new = covariates_interp_aux(coords_pad)
    else:
        x_cov_new, x_cov_var_new = build_grid_covariates(ckpt, coords_pad)
    ckpt["model"].eval()  # predict_on_grid_attached leaves the model in train mode
    (g, gs), (m, ms) = predict_on_grid_attached(
        model=ckpt["model"],
        coords_stations=coords_union, x_cov_stations=x_cov, x_cov_var_stations=x_cov_var,
        coords_grid=coords_pad, x_cov_grid=x_cov_new, x_cov_var_grid=x_cov_var_new,
        X_stats_pred=X_stats, X_stats_graph=Xg_stats, stats_targets=stats,
        k_graph=K_GRAPH, n_mc=N_MC, pe_num_freqs_graph=NUM_FREQS, device=device,
    )
    mean, std = ((g, gs) if est_grav else (m, ms))
    mean, std = mean[:-2], std[:-2]
    # outputs are in the [-1, 1] training scale -> physical units
    return unnormalization(mean, z_min, z_max), std * 0.5 * (z_max - z_min)


gx, gy = np.mgrid[
    coords_st[:, 0].min():coords_st[:, 0].max():complex(0, NX),
    coords_st[:, 1].min():coords_st[:, 1].max():complex(0, NY),
]
grid_coords = np.column_stack([gx.ravel(), gy.ravel()])
grid_model, grid_std = (a.reshape(NX, NY) for a in predict_points(grid_coords))

# residual correction so the map honors the stations and keeps the peaks
pred_st_attached, _ = predict_points(coords_st)
use = np.ones(len(z_st_raw), bool) if RESIDUALS_FROM == "all" else mask_train_target
resid = z_st_raw[use] - pred_st_attached[use]
resid_grid = griddata(coords_st[use], resid, (gx, gy), method="linear")
grid_final = grid_model + np.nan_to_num(resid_grid, nan=0.0)
print(f"Rango malla modelo: [{grid_model.min():.3f}, {grid_model.max():.3f}] {units}")
print(f"Rango malla final : [{grid_final.min():.3f}, {grid_final.max():.3f}] {units}")

# clip to the station convex hull (outside it the model only extrapolates)
inside = (Delaunay(coords_st).find_simplex(grid_coords) >= 0).reshape(NX, NY)

# ---- 6. save ----
out_base = f"lsgl_pipeline_grid_{tag}_{NX}x{NY}"
np.savez(f"{OUT_DATA}/{out_base}.npz", grid_x=gx, grid_y=gy, grid_z_model=grid_model,
         grid_z_final=grid_final, grid_std=grid_std, inside_hull=inside, units=units, label_var=TARGET,
         rmse_test=rmse_test, idx_test=idx_test)
pd.DataFrame({
    "X": gx.ravel(), "Y": gy.ravel(), "modelo": grid_model.ravel(), "final": grid_final.ravel(),
    "std": grid_std.ravel(), "dentro_envolvente": inside.ravel(),
}).to_csv(f"{OUT_DATA}/{out_base}.csv", index=False)
print(f"Malla: {OUT_DATA}/{out_base}.npz / .csv")

panels = [
    (grid_final, f"Mapa final {TARGET} (grafos + residuos) -- {NX}x{NY}", "final", CBAR_LABEL[TARGET], "jet", (z_min, z_max)),
    (grid_model, f"Predicción del modelo de grafos {TARGET} -- {NX}x{NY}", "modelo", CBAR_LABEL[TARGET], "jet", (z_min, z_max)),
    (grid_std, f"Incertidumbre (std MC-dropout) {TARGET}", "std", f"Desviación estándar ({units})", "viridis", (None, None)),
]
for grid, title, suffix, cbar, cmap, (vmin, vmax) in panels:
    fig, ax = plt.subplots(figsize=(8, 6.5))
    mesh = ax.pcolormesh(gx, gy, np.where(inside, grid, np.nan), shading="auto", cmap=cmap, vmin=vmin, vmax=vmax)
    fig.colorbar(mesh, ax=ax, label=cbar)
    if suffix == "std":
        ax.scatter(*coords_st.T, c="white", edgecolors="black", linewidths=0.6, s=20, label="Estaciones")
    else:
        ax.scatter(*coords_st[mask_train_target].T, c=z_st_raw[mask_train_target], cmap=cmap, vmin=vmin, vmax=vmax,
                   edgecolors="black", linewidths=0.6, s=24, label="Train (valor real)")
    test_color = dict(c="white") if suffix == "std" else dict(c=z_st_raw[idx_test], cmap=cmap, vmin=vmin, vmax=vmax)
    ax.scatter(*coords_st[idx_test].T, **test_color, marker="^", edgecolors="magenta", linewidths=1.4, s=60,
               label=f"Test (RMSE={rmse_test:.2f} {units})")
    ax.set_xlabel("Este (m)")
    ax.set_ylabel("Norte (m)")
    ax.set_title(title)
    ax.set_aspect("equal", adjustable="box")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(f"{OUT_FIG}/{out_base}_{suffix}.png", dpi=300, bbox_inches="tight")
    fig.savefig(f"{OUT_FIG}/{out_base}_{suffix}.svg", bbox_inches="tight")
    plt.close(fig)
print(f"Figuras: {OUT_FIG}/{out_base}_{{final,modelo,std}}.png / .svg")
