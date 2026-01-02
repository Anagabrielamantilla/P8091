import torch, math
import numpy as np

def grav3D_8091(density_contrast_model, K: torch.Tensor, to_mgal: bool = True):
    def _ensure_tensor_local(x, device="cpu", dtype=torch.float32):
        if isinstance(x, torch.Tensor):
            return x.to(device=device, dtype=dtype)
        return torch.as_tensor(x, device=device, dtype=dtype)

    if not isinstance(K, torch.Tensor):
        raise TypeError("K debe ser un torch.Tensor.")

    device = K.device
    dtype  = K.dtype

    rho = _ensure_tensor_local(density_contrast_model, device=device, dtype=dtype).reshape(-1)

    valid_mask = ~torch.isnan(rho)
    if not valid_mask.any():
        raise RuntimeError("El modelo no tiene celdas válidas (todas son NaN).")

    rho_valid = rho[valid_mask]

    if K.shape[1] != rho_valid.numel():
        raise ValueError(
            f"K tiene {K.shape[1]} columnas pero el modelo válido tiene {rho_valid.numel()} celdas. "
            "Asegúrate de que K se haya construido con el mismo patrón de NaN."
        )

    gz = (K @ rho_valid)  # m/s^2
    return gz * 1e5 if to_mgal else gz
