import torch, math
import numpy as np

def get_B0_and_m(model: np.ndarray,
                 device: torch.device,
                 dtype: torch.dtype,
                 I_deg: float =90.0,
                 D_deg: float=0.0,
                 B0_nT: float =50000.0):

    model_t = torch.from_numpy(model).to(device, dtype)

    active = ~torch.isnan(model_t) & (model_t != 0)
    k_act = model_t[active]

    B0_mag = torch.as_tensor(B0_nT * 1e-9, device=device, dtype=dtype)

    I = torch.deg2rad(torch.as_tensor(I_deg, device=device, dtype=dtype))
    D = torch.deg2rad(torch.as_tensor(D_deg, device=device, dtype=dtype))

    B0_vec = B0_mag * torch.stack([
        torch.cos(I) * torch.sin(D),
        torch.cos(I) * torch.cos(D),
        -torch.sin(I)
    ])

    B0_unit = B0_vec / B0_vec.norm()

    # ---- magnetización ----
    import math
    mu0 = torch.as_tensor(4.0 * math.pi * 1e-7, device=device, dtype=dtype)
    m = k_act * (B0_mag / mu0)

    return B0_vec, B0_unit, B0_mag, m


def compute_kernel(cell_centers: np.ndarray,
                   receiver_location: np.ndarray,
                   B0_unit: torch.Tensor,
                   cell_volume: float,
                   model: np.ndarray,
                   device: torch.device,
                   dtype: torch.dtype):
  
    cell_centers_t = torch.from_numpy(cell_centers).to(device, dtype)   # (nC,3)
    receiver_location_t = torch.from_numpy(receiver_location).to(device, dtype)
    model_t = torch.from_numpy(model).to(device, dtype)

    active = ~torch.isnan(model_t) & (model_t != 0)
    cc_act  = cell_centers_t[active]
    vol_act = torch.full((cc_act.shape[0],), cell_volume, device=device, dtype=dtype) # Correct way to get volume for active cells

    mu0 = torch.as_tensor(4.0 * math.pi * 1e-7, device=device, dtype=dtype)
    cm = -mu0 / (4.0 * math.pi)

    rx = cc_act[:,0].unsqueeze(0) - receiver_location_t[:,0].unsqueeze(1)
    ry = cc_act[:,1].unsqueeze(0) - receiver_location_t[:,1].unsqueeze(1)
    rz = cc_act[:,2].unsqueeze(0) - receiver_location_t[:,2].unsqueeze(1)

    r2 = rx*rx + ry*ry + rz*rz
    r  = torch.sqrt(r2)
    r3 = r2 * r
    r5 = r3 * r2

    B0dotr = B0_unit[0]*rx + B0_unit[1]*ry + B0_unit[2]*rz

    kernel_core = (1.0/r3) - 3.0*(B0dotr**2)/r5

    kernel = cm * vol_act.unsqueeze(0) * kernel_core

    return kernel

# Kernel gravimetria

def calculateKernelGrav(density_contrast_model, mesh, receiver_locations) -> torch.Tensor:
    def _ensure_tensor_local(x, device="cpu", dtype=torch.float32):
        if isinstance(x, torch.Tensor):
            return x.to(device=device, dtype=dtype)
        return torch.as_tensor(x, device=device, dtype=dtype)

    if isinstance(density_contrast_model, torch.Tensor):
        device = density_contrast_model.device
        dtype  = density_contrast_model.dtype
    else:
        device = "cpu"
        dtype  = torch.float32

    rho = _ensure_tensor_local(density_contrast_model, device=device, dtype=dtype).reshape(-1)

    valid_mask = ~torch.isnan(rho)
    if not valid_mask.any():
        raise RuntimeError("El modelo no tiene celdas válidas (todas son NaN).")

    centers = _ensure_tensor_local(mesh["cell_centers"], device=device, dtype=dtype)
    centers_v = centers[valid_mask]

    obs = _ensure_tensor_local(receiver_locations, device=device, dtype=dtype)
    if obs.ndim != 2:
        raise ValueError("receiver_locations debe ser 2D.")
    if obs.shape[1] == 2:
        obs_xyz = torch.cat([obs, torch.zeros((obs.shape[0], 1), device=device, dtype=dtype)], dim=1)
    elif obs.shape[1] == 3:
        obs_xyz = obs
    else:
        raise ValueError("receiver_locations debe tener 2 o 3 columnas.")

    G = torch.as_tensor(6.67430e-11, device=device, dtype=dtype)

    dx = _ensure_tensor_local(mesh["dx"], device=device, dtype=dtype)
    dy = _ensure_tensor_local(mesh["dy"], device=device, dtype=dtype)
    dz = _ensure_tensor_local(mesh["dz"], device=device, dtype=dtype)

    nC  = centers.shape[0]
    nCv = centers_v.shape[0]

    def _select_sizes(h):
        if h.ndim == 0:
            return h.expand(nCv)
        if h.ndim == 1 and h.numel() == nC:
            return h[valid_mask]
        if h.ndim == 1 and h.numel() == nCv:
            return h
        raise ValueError("dx/dy/dz deben ser escalares o vectores tamaño nC (por celda).")

    dx_v = _select_sizes(dx)
    dy_v = _select_sizes(dy)
    dz_v = _select_sizes(dz)

    V_v = dx_v * dy_v * dz_v
    GV8 = G * (V_v / 8.0)

    qx, qy, qz = dx_v / 4.0, dy_v / 4.0, dz_v / 4.0

    signs = torch.tensor(
        [[+1,+1,+1],[+1,+1,-1],[+1,-1,+1],[+1,-1,-1],
         [-1,+1,+1],[-1,+1,-1],[-1,-1,+1],[-1,-1,-1]],
        device=device, dtype=dtype
    )

    Xc = centers_v[:, 0].unsqueeze(0)
    Yc = centers_v[:, 1].unsqueeze(0)
    Zc = centers_v[:, 2].unsqueeze(0)

    sx = obs_xyz[:, 0].unsqueeze(1)
    sy = obs_xyz[:, 1].unsqueeze(1)
    sz = obs_xyz[:, 2].unsqueeze(1)

    K = torch.zeros((obs_xyz.shape[0], nCv), device=device, dtype=dtype)

    for s in range(8):
        ox = (signs[s, 0] * qx).unsqueeze(0)
        oy = (signs[s, 1] * qy).unsqueeze(0)
        oz = (signs[s, 2] * qz).unsqueeze(0)

        Xp = Xc + ox
        Yp = Yc + oy
        Zp = Zc + oz

        dx_ = sx - Xp
        dy_ = sy - Yp
        dz_ = Zp - sz  # Z positivo hacia arriba

        r2 = dx_*dx_ + dy_*dy_ + dz_*dz_
        inv_r3 = torch.pow(r2, -1.5)

        K = K + (GV8.unsqueeze(0) * dz_ * inv_r3)

    return K

