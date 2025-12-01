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
