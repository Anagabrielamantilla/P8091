import math
import numpy as np
import scipy.sparse.linalg as spla
import torch
import discretize
import scipy.sparse as sp
from discretize.utils import volume_average

from Forward.solvers import make_solver
from geoana.kernels import prism_fzz, prism_fzx, prism_fzy
from geoana.kernels import potential_field_prism as _pfp
import numpy as np


def geomagnetic_field(
    I_deg: float,
    D_deg: float,
    amplitude_nT: float,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float64,
):
    
    """
    Construye el vector del campo geomagnético principal B0 a partir de
    inclinación (I), declinación (D) y amplitud, y retorna también su dirección
    unitaria.

    Parámetros
    ----------
    I_deg : float
        Inclinación en grados.
    D_deg : float
        Declinación en grados.
    amplitude_nT : float
        Magnitud del campo (en nT).
    device, dtype :
        Dispositivo y tipo numérico para los tensores.

    Retorna
    -------
    B0_vec_nT : torch.Tensor (3,)
        Vector B0 en nT: [Bx, By, Bz].
    B0_unit : torch.Tensor (3,)
        Vector unitario en la dirección de B0.
    """

    device = torch.device(device)
    I = torch.deg2rad(torch.tensor(I_deg, device=device, dtype=dtype))
    D = torch.deg2rad(torch.tensor(D_deg, device=device, dtype=dtype))

    B0_vec_nT = torch.tensor(float(amplitude_nT), device=device, dtype=dtype) * torch.stack(
        [
            torch.cos(I) * torch.sin(D),   # x
            torch.cos(I) * torch.cos(D),   # y
            -torch.sin(I),                 # z
        ]
    )

    B0_unit = B0_vec_nT / torch.linalg.norm(B0_vec_nT)
    return B0_vec_nT, B0_unit

def calculateKernelMag(
    model,
    cell_centers,
    receiver_locations,
    dx, dy, dz,
    B0_vec_nT,
    B0_unit,
    chunk_obs: int = 64,
):
    
    """
    Calcula el kernel (matriz de sensibilidad) magnético para un conjunto de
    prismas rectangulares 3D, usando integrales analíticas tipo “prisma”
    (prism_fzz, prism_fzx, prism_fzy) y sumas por inclusión–exclusión en las 8
    esquinas del prisma.

    Este kernel implementa una forma estándar para respuesta tipo “campo total”
    proyectada en la dirección B0_unit, con una dependencia de amplitud dada por
    B0_vec_nT. Las unidades del resultado quedan coherentes con las unidades de
    B0_vec_nT y con la convención de las funciones prism_* (incluye el factor
    1/(4π) tal como está en el código).

    Parámetros
    ----------
    model : array-like o torch.Tensor (nC,)
        Susceptibilidad (o parámetro escalar) por celda. NaN/inf -> celda inactiva.
    cell_centers : array-like o torch.Tensor (nC,3)
        Centros (x,y,z) de cada celda.
    receiver_locations : array-like o torch.Tensor (nObs,2) o (nObs,3)
        Receptores; si es 2D se asume z=0.
    dx, dy, dz : escalar o vector
        Tamaños de celda. Pueden ser:
        - escalares,
        - vectores tamaño nC (malla completa), o
        - vectores tamaño nCv (ya filtrados a activas).
    B0_vec_nT : array-like o torch.Tensor (3,)
        Vector B0 (no unitario), típicamente en nT.
    B0_unit : array-like o torch.Tensor (3,)
        Dirección unitaria de B0.
    chunk_obs : int
        Número de observaciones procesadas por bloque (controla memoria).

    Retorna
    -------
    K : torch.Tensor (nObs, nCv)
        Kernel para celdas activas (columnas = celdas activas, filas = receptores).
    chi_active : torch.Tensor (nCv,)
        Valores del modelo filtrados a celdas activas, en el mismo orden de K.
    """
    
    def _ensure_tensor_local(x, device, dtype):
        if isinstance(x, torch.Tensor):
            return x.to(device=device, dtype=dtype)
        return torch.as_tensor(x, device=device, dtype=dtype)

    if isinstance(model, torch.Tensor):
        device = model.device
        dtype = model.dtype
    elif isinstance(B0_unit, torch.Tensor):
        device = B0_unit.device
        dtype = B0_unit.dtype
    else:
        device = torch.device("cpu")
        dtype = torch.float64

    model_t = _ensure_tensor_local(model, device=device, dtype=dtype).reshape(-1)         # (nC,)
    cc_t    = _ensure_tensor_local(cell_centers, device=device, dtype=dtype).reshape(-1, 3)  # (nC,3)

    obs_t = _ensure_tensor_local(receiver_locations, device=device, dtype=dtype)
    if obs_t.ndim != 2:
        raise ValueError("receiver_locations debe ser 2D")
    if obs_t.shape[1] == 2:
        obs_t = torch.cat(
            [obs_t, torch.zeros((obs_t.shape[0], 1), device=device, dtype=dtype)],
            dim=1
        )
    elif obs_t.shape[1] != 3:
        raise ValueError("receiver_locations debe tener 2 o 3 columnas")

    dx_t = _ensure_tensor_local(dx, device=device, dtype=dtype)
    dy_t = _ensure_tensor_local(dy, device=device, dtype=dtype)
    dz_t = _ensure_tensor_local(dz, device=device, dtype=dtype)

    b0_vec_t  = _ensure_tensor_local(B0_vec_nT, device=device, dtype=dtype).reshape(3)
    b0_unit_t = _ensure_tensor_local(B0_unit,   device=device, dtype=dtype).reshape(3)

    active = torch.isfinite(model_t)
    if not active.any():
        raise RuntimeError("El modelo no tiene celdas activas (todas NaN/inf).")

    cc_v       = cc_t[active]     # (nCv,3)
    chi_active = model_t[active]  # (nCv,)

    nC  = model_t.numel()
    nCv = chi_active.numel()
    nObs = obs_t.shape[0]

    def _select_sizes(h):

        if isinstance(h, torch.Tensor) and h.ndim > 0:
            h1 = h.reshape(-1)
        else:
            h1 = h

        if h1.ndim == 0:
            return h1.expand(nCv)
        if h1.ndim == 1 and h1.numel() == nC:
            return h1[active]
        if h1.ndim == 1 and h1.numel() == nCv:
            return h1
        raise ValueError("dx/dy/dz deben ser escalares o vectores tamaño nC (o nCv).")

    dx_v = _select_sizes(dx_t)
    dy_v = _select_sizes(dy_t)
    dz_v = _select_sizes(dz_t)

    hx = dx_v / 2.0
    hy = dy_v / 2.0
    hz = dz_v / 2.0

    corners = torch.tensor(
        [[-1,-1,-1],[-1,-1, 1],[-1, 1,-1],[-1, 1, 1],
         [ 1,-1,-1],[ 1,-1, 1],[ 1, 1,-1],[ 1, 1, 1]],
        device=device, dtype=dtype
    )
    alt = torch.tensor([ 1, -1, -1,  1, -1,  1,  1, -1], device=device, dtype=dtype)

    Xc = cc_v[:, 0]
    Yc = cc_v[:, 1]
    Zc = cc_v[:, 2]

    bx, by, bz = b0_unit_t
    Mx, My, Mz = b0_vec_t

    K = torch.empty((nObs, nCv), device=device, dtype=dtype)

    for i0 in range(0, nObs, chunk_obs):
        i1 = min(i0 + chunk_obs, nObs)
        obs_c = obs_t[i0:i1, :]  # (cObs,3)
        cObs = obs_c.shape[0]

        sx = obs_c[:, 0].detach().cpu().numpy().reshape(cObs, 1)
        sy = obs_c[:, 1].detach().cpu().numpy().reshape(cObs, 1)
        sz = obs_c[:, 2].detach().cpu().numpy().reshape(cObs, 1)

        Xc_np = Xc.detach().cpu().numpy().reshape(1, nCv)
        Yc_np = Yc.detach().cpu().numpy().reshape(1, nCv)
        Zc_np = Zc.detach().cpu().numpy().reshape(1, nCv)

        hx_np = hx.detach().cpu().numpy().reshape(1, nCv)
        hy_np = hy.detach().cpu().numpy().reshape(1, nCv)
        hz_np = hz.detach().cpu().numpy().reshape(1, nCv)

        gxx = np.zeros((cObs, nCv), dtype=np.float64)
        gxy = np.zeros((cObs, nCv), dtype=np.float64)
        gxz = np.zeros((cObs, nCv), dtype=np.float64)
        gyy = np.zeros((cObs, nCv), dtype=np.float64)
        gyz = np.zeros((cObs, nCv), dtype=np.float64)
        gzz = np.zeros((cObs, nCv), dtype=np.float64)

        bx_f = float(bx.item()); by_f = float(by.item()); bz_f = float(bz.item())
        Mx_f = float(Mx.item()); My_f = float(My.item()); Mz_f = float(Mz.item())

        for k in range(8):
            ox = float(corners[k, 0].item()) * hx_np
            oy = float(corners[k, 1].item()) * hy_np
            oz = float(corners[k, 2].item()) * hz_np
            sgn = float(alt[k].item())

            dxn = (Xc_np + ox) - sx
            dyn = (Yc_np + oy) - sy
            dzn = (Zc_np + oz) - sz

            gxx += sgn * prism_fzz(dyn, dzn, dxn)
            gxy += sgn * prism_fzx(dyn, dzn, dxn)
            gxz += sgn * prism_fzy(dyn, dzn, dxn)

            gyy += sgn * prism_fzz(dzn, dxn, dyn)
            gyz += sgn * prism_fzy(dxn, dyn, dzn)
            gzz += sgn * prism_fzz(dxn, dyn, dzn)

        vals_x = bx_f*gxx + by_f*gxy + bz_f*gxz
        vals_y = bx_f*gxy + by_f*gyy + bz_f*gyz
        vals_z = bx_f*gxz + by_f*gyz + bz_f*gzz

        cell_vals = vals_x*Mx_f + vals_y*My_f + vals_z*Mz_f
        K_chunk = cell_vals / (4.0 * math.pi)

        K[i0:i1, :] = torch.from_numpy(K_chunk).to(device=device, dtype=dtype)

    return K, chi_active


def calculateKernelGrav(
    density_contrast_model,
    mesh,
    receiver_locations,
    chunk_cells: int = 2000,
) -> torch.Tensor:

    """
    Calcula el kernel (matriz de sensibilidad) de gravimetría para una malla 3D de
    prismas rectangulares usando la formulación exacta de GeoAna.

    Parámetros
    ----------
    density_contrast_model : array-like o torch.Tensor (nC,)
        Contraste de densidad por celda; NaN = celda inactiva.
    mesh : dict
        Debe incluir: "cell_centers" (nC,3) y "dx","dy","dz" (escalares o vectores).
    receiver_locations : array-like o torch.Tensor (nObs,2) o (nObs,3)
        Coordenadas de receptores; si es 2D se asume z=0.
    chunk_cells : int
        Número de celdas activas por bloque para controlar memoria.

    Retorna
    -------
    K : torch.Tensor (nObs, nCv)
        Kernel para las celdas activas.
    centers_v : torch.Tensor (nCv,3)
        Centros de las celdas activas.
    """

    def _ensure_tensor_local(x, device="cpu", dtype=torch.float32):
        if isinstance(x, torch.Tensor):
            return x.to(device=device, dtype=dtype)
        return torch.as_tensor(x, device=device, dtype=dtype)

    # --- device/dtype coherentes con el modelo
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
    centers_v = centers[valid_mask]  # solo celdas activas

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

    # --- prism bounds por celda (en numpy)
    Xc_np = centers_v[:, 0].detach().cpu().numpy()
    Yc_np = centers_v[:, 1].detach().cpu().numpy()
    Zc_np = centers_v[:, 2].detach().cpu().numpy()

    dx_np = dx_v.detach().cpu().numpy()
    dy_np = dy_v.detach().cpu().numpy()
    dz_np = dz_v.detach().cpu().numpy()

    x1 = Xc_np - dx_np / 2.0
    x2 = Xc_np + dx_np / 2.0
def geomagnetic_field(
    I_deg: float,
    D_deg: float,
    amplitude_nT: float,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float64,
):
    
    """
    Construye el vector del campo geomagnético principal B0 a partir de
    inclinación (I), declinación (D) y amplitud, y retorna también su dirección
    unitaria.

    Parámetros
    ----------
    I_deg : float
        Inclinación en grados.
    D_deg : float
        Declinación en grados.
    amplitude_nT : float
        Magnitud del campo (en nT).
    device, dtype :
        Dispositivo y tipo numérico para los tensores.

    Retorna
    -------
    B0_vec_nT : torch.Tensor (3,)
        Vector B0 en nT: [Bx, By, Bz].
    B0_unit : torch.Tensor (3,)
        Vector unitario en la dirección de B0.
    """

    device = torch.device(device)
    I = torch.deg2rad(torch.tensor(I_deg, device=device, dtype=dtype))
    D = torch.deg2rad(torch.tensor(D_deg, device=device, dtype=dtype))

    B0_vec_nT = torch.tensor(float(amplitude_nT), device=device, dtype=dtype) * torch.stack(
        [
            torch.cos(I) * torch.sin(D),   # x
            torch.cos(I) * torch.cos(D),   # y
            -torch.sin(I),                 # z
        ]
    )

    B0_unit = B0_vec_nT / torch.linalg.norm(B0_vec_nT)
    return B0_vec_nT, B0_unit

def calculateKernelMag(
    model,
    cell_centers,
    receiver_locations,
    dx, dy, dz,
    B0_vec_nT,
    B0_unit,
    chunk_obs: int = 64,
):
    
    """
    Calcula el kernel (matriz de sensibilidad) magnético para un conjunto de
    prismas rectangulares 3D, usando integrales analíticas tipo “prisma”
    (prism_fzz, prism_fzx, prism_fzy) y sumas por inclusión–exclusión en las 8
    esquinas del prisma.

    Este kernel implementa una forma estándar para respuesta tipo “campo total”
    proyectada en la dirección B0_unit, con una dependencia de amplitud dada por
    B0_vec_nT. Las unidades del resultado quedan coherentes con las unidades de
    B0_vec_nT y con la convención de las funciones prism_* (incluye el factor
    1/(4π) tal como está en el código).

    Parámetros
    ----------
    model : array-like o torch.Tensor (nC,)
        Susceptibilidad (o parámetro escalar) por celda. NaN/inf -> celda inactiva.
    cell_centers : array-like o torch.Tensor (nC,3)
        Centros (x,y,z) de cada celda.
    receiver_locations : array-like o torch.Tensor (nObs,2) o (nObs,3)
        Receptores; si es 2D se asume z=0.
    dx, dy, dz : escalar o vector
        Tamaños de celda. Pueden ser:
        - escalares,
        - vectores tamaño nC (malla completa), o
        - vectores tamaño nCv (ya filtrados a activas).
    B0_vec_nT : array-like o torch.Tensor (3,)
        Vector B0 (no unitario), típicamente en nT.
    B0_unit : array-like o torch.Tensor (3,)
        Dirección unitaria de B0.
    chunk_obs : int
        Número de observaciones procesadas por bloque (controla memoria).

    Retorna
    -------
    K : torch.Tensor (nObs, nCv)
        Kernel para celdas activas (columnas = celdas activas, filas = receptores).
    chi_active : torch.Tensor (nCv,)
        Valores del modelo filtrados a celdas activas, en el mismo orden de K.
    """
    
    def _ensure_tensor_local(x, device, dtype):
        if isinstance(x, torch.Tensor):
            return x.to(device=device, dtype=dtype)
        return torch.as_tensor(x, device=device, dtype=dtype)

    if isinstance(model, torch.Tensor):
        device = model.device
        dtype = model.dtype
    elif isinstance(B0_unit, torch.Tensor):
        device = B0_unit.device
        dtype = B0_unit.dtype
    else:
        device = torch.device("cpu")
        dtype = torch.float64

    model_t = _ensure_tensor_local(model, device=device, dtype=dtype).reshape(-1)         # (nC,)
    cc_t    = _ensure_tensor_local(cell_centers, device=device, dtype=dtype).reshape(-1, 3)  # (nC,3)

    obs_t = _ensure_tensor_local(receiver_locations, device=device, dtype=dtype)
    if obs_t.ndim != 2:
        raise ValueError("receiver_locations debe ser 2D")
    if obs_t.shape[1] == 2:
        obs_t = torch.cat(
            [obs_t, torch.zeros((obs_t.shape[0], 1), device=device, dtype=dtype)],
            dim=1
        )
    elif obs_t.shape[1] != 3:
        raise ValueError("receiver_locations debe tener 2 o 3 columnas")

    dx_t = _ensure_tensor_local(dx, device=device, dtype=dtype)
    dy_t = _ensure_tensor_local(dy, device=device, dtype=dtype)
    dz_t = _ensure_tensor_local(dz, device=device, dtype=dtype)

    b0_vec_t  = _ensure_tensor_local(B0_vec_nT, device=device, dtype=dtype).reshape(3)
    b0_unit_t = _ensure_tensor_local(B0_unit,   device=device, dtype=dtype).reshape(3)

    active = torch.isfinite(model_t)
    if not active.any():
        raise RuntimeError("El modelo no tiene celdas activas (todas NaN/inf).")

    cc_v       = cc_t[active]     # (nCv,3)
    chi_active = model_t[active]  # (nCv,)

    nC  = model_t.numel()
    nCv = chi_active.numel()
    nObs = obs_t.shape[0]

    def _select_sizes(h):

        if isinstance(h, torch.Tensor) and h.ndim > 0:
            h1 = h.reshape(-1)
        else:
            h1 = h

        if h1.ndim == 0:
            return h1.expand(nCv)
        if h1.ndim == 1 and h1.numel() == nC:
            return h1[active]
        if h1.ndim == 1 and h1.numel() == nCv:
            return h1
        raise ValueError("dx/dy/dz deben ser escalares o vectores tamaño nC (o nCv).")

    dx_v = _select_sizes(dx_t)
    dy_v = _select_sizes(dy_t)
    dz_v = _select_sizes(dz_t)

    hx = dx_v / 2.0
    hy = dy_v / 2.0
    hz = dz_v / 2.0

    corners = torch.tensor(
        [[-1,-1,-1],[-1,-1, 1],[-1, 1,-1],[-1, 1, 1],
         [ 1,-1,-1],[ 1,-1, 1],[ 1, 1,-1],[ 1, 1, 1]],
        device=device, dtype=dtype
    )
    alt = torch.tensor([ 1, -1, -1,  1, -1,  1,  1, -1], device=device, dtype=dtype)

    Xc = cc_v[:, 0]
    Yc = cc_v[:, 1]
    Zc = cc_v[:, 2]

    bx, by, bz = b0_unit_t
    Mx, My, Mz = b0_vec_t

    K = torch.empty((nObs, nCv), device=device, dtype=dtype)

    for i0 in range(0, nObs, chunk_obs):
        i1 = min(i0 + chunk_obs, nObs)
        obs_c = obs_t[i0:i1, :]  # (cObs,3)
        cObs = obs_c.shape[0]

        sx = obs_c[:, 0].detach().cpu().numpy().reshape(cObs, 1)
        sy = obs_c[:, 1].detach().cpu().numpy().reshape(cObs, 1)
        sz = obs_c[:, 2].detach().cpu().numpy().reshape(cObs, 1)

        Xc_np = Xc.detach().cpu().numpy().reshape(1, nCv)
        Yc_np = Yc.detach().cpu().numpy().reshape(1, nCv)
        Zc_np = Zc.detach().cpu().numpy().reshape(1, nCv)

        hx_np = hx.detach().cpu().numpy().reshape(1, nCv)
        hy_np = hy.detach().cpu().numpy().reshape(1, nCv)
        hz_np = hz.detach().cpu().numpy().reshape(1, nCv)

        gxx = np.zeros((cObs, nCv), dtype=np.float64)
        gxy = np.zeros((cObs, nCv), dtype=np.float64)
        gxz = np.zeros((cObs, nCv), dtype=np.float64)
        gyy = np.zeros((cObs, nCv), dtype=np.float64)
        gyz = np.zeros((cObs, nCv), dtype=np.float64)
        gzz = np.zeros((cObs, nCv), dtype=np.float64)

        bx_f = float(bx.item()); by_f = float(by.item()); bz_f = float(bz.item())
        Mx_f = float(Mx.item()); My_f = float(My.item()); Mz_f = float(Mz.item())

        for k in range(8):
            ox = float(corners[k, 0].item()) * hx_np
            oy = float(corners[k, 1].item()) * hy_np
            oz = float(corners[k, 2].item()) * hz_np
            sgn = float(alt[k].item())

            dxn = (Xc_np + ox) - sx
            dyn = (Yc_np + oy) - sy
            dzn = (Zc_np + oz) - sz

            gxx += sgn * prism_fzz(dyn, dzn, dxn)
            gxy += sgn * prism_fzx(dyn, dzn, dxn)
            gxz += sgn * prism_fzy(dyn, dzn, dxn)

            gyy += sgn * prism_fzz(dzn, dxn, dyn)
            gyz += sgn * prism_fzy(dxn, dyn, dzn)
            gzz += sgn * prism_fzz(dxn, dyn, dzn)

        vals_x = bx_f*gxx + by_f*gxy + bz_f*gxz
        vals_y = bx_f*gxy + by_f*gyy + bz_f*gyz
        vals_z = bx_f*gxz + by_f*gyz + bz_f*gzz

        cell_vals = vals_x*Mx_f + vals_y*My_f + vals_z*Mz_f
        K_chunk = cell_vals / (4.0 * math.pi)

        K[i0:i1, :] = torch.from_numpy(K_chunk).to(device=device, dtype=dtype)

    return K, chi_active


def calculateKernelGrav(
    density_contrast_model,
    mesh,
    receiver_locations,
    chunk_cells: int = 2000,
) -> torch.Tensor:

    """
    Calcula el kernel (matriz de sensibilidad) de gravimetría para una malla 3D de
    prismas rectangulares usando la formulación exacta de GeoAna.

    Parámetros
    ----------
    density_contrast_model : array-like o torch.Tensor (nC,)
        Contraste de densidad por celda; NaN = celda inactiva.
    mesh : dict
        Debe incluir: "cell_centers" (nC,3) y "dx","dy","dz" (escalares o vectores).
    receiver_locations : array-like o torch.Tensor (nObs,2) o (nObs,3)
        Coordenadas de receptores; si es 2D se asume z=0.
    chunk_cells : int
        Número de celdas activas por bloque para controlar memoria.

    Retorna
    -------
    K : torch.Tensor (nObs, nCv)
        Kernel para las celdas activas.
    centers_v : torch.Tensor (nCv,3)
        Centros de las celdas activas.
    """

    def _ensure_tensor_local(x, device="cpu", dtype=torch.float32):
        if isinstance(x, torch.Tensor):
            return x.to(device=device, dtype=dtype)
        return torch.as_tensor(x, device=device, dtype=dtype)

    # --- device/dtype coherentes con el modelo
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
    centers_v = centers[valid_mask]  # solo celdas activas

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

    # --- prism bounds por celda (en numpy)
    Xc_np = centers_v[:, 0].detach().cpu().numpy()
    Yc_np = centers_v[:, 1].detach().cpu().numpy()
    Zc_np = centers_v[:, 2].detach().cpu().numpy()

    dx_np = dx_v.detach().cpu().numpy()
    dy_np = dy_v.detach().cpu().numpy()
    dz_np = dz_v.detach().cpu().numpy()

    x1 = Xc_np - dx_np / 2.0
    x2 = Xc_np + dx_np / 2.0
    y1 = Yc_np - dy_np / 2.0
    y2 = Yc_np + dy_np / 2.0
    z1 = Zc_np - dz_np / 2.0
    z2 = Zc_np + dz_np / 2.0

    sx = obs_xyz[:, 0].detach().cpu().numpy()
    sy = obs_xyz[:, 1].detach().cpu().numpy()
    sz = obs_xyz[:, 2].detach().cpu().numpy()

    K = torch.zeros((obs_xyz.shape[0], nCv), device=device, dtype=dtype)
    G_np = float(G.detach().cpu().numpy())

    # --- exact prism kernel (geoana) por chunks
    for i0 in range(0, nCv, chunk_cells):
        i1 = min(i0 + chunk_cells, nCv)

        x1d = x1[i0:i1][None, :] - sx[:, None]
        x2d = x2[i0:i1][None, :] - sx[:, None]
        y1d = y1[i0:i1][None, :] - sy[:, None]
        y2d = y2[i0:i1][None, :] - sy[:, None]
        z1d = z1[i0:i1][None, :] - sz[:, None]
        z2d = z2[i0:i1][None, :] - sz[:, None]

        term = (
            _pfp.prism_fz(x2d, y2d, z2d)
            - _pfp.prism_fz(x1d, y2d, z2d)
            - _pfp.prism_fz(x2d, y1d, z2d)
            + _pfp.prism_fz(x1d, y1d, z2d)
            - _pfp.prism_fz(x2d, y2d, z1d)
            + _pfp.prism_fz(x1d, y2d, z1d)
            + _pfp.prism_fz(x2d, y1d, z1d)
            - _pfp.prism_fz(x1d, y1d, z1d)
        )

        K_chunk = (G_np * term).astype(np.float64, copy=False)
        K[:, i0:i1] = torch.from_numpy(K_chunk).to(device=device, dtype=dtype)

    return K, centers_v
    y1 = Yc_np - dy_np / 2.0
    y2 = Yc_np + dy_np / 2.0
    z1 = Zc_np - dz_np / 2.0
    z2 = Zc_np + dz_np / 2.0

    sx = obs_xyz[:, 0].detach().cpu().numpy()
    sy = obs_xyz[:, 1].detach().cpu().numpy()
    sz = obs_xyz[:, 2].detach().cpu().numpy()

    K = torch.zeros((obs_xyz.shape[0], nCv), device=device, dtype=dtype)
    G_np = float(G.detach().cpu().numpy())

    # --- exact prism kernel (geoana) por chunks
    for i0 in range(0, nCv, chunk_cells):
        i1 = min(i0 + chunk_cells, nCv)

        x1d = x1[i0:i1][None, :] - sx[:, None]
        x2d = x2[i0:i1][None, :] - sx[:, None]
        y1d = y1[i0:i1][None, :] - sy[:, None]
        y2d = y2[i0:i1][None, :] - sy[:, None]
        z1d = z1[i0:i1][None, :] - sz[:, None]
        z2d = z2[i0:i1][None, :] - sz[:, None]

        term = (
            _pfp.prism_fz(x2d, y2d, z2d)
            - _pfp.prism_fz(x1d, y2d, z2d)
            - _pfp.prism_fz(x2d, y1d, z2d)
            + _pfp.prism_fz(x1d, y1d, z2d)
            - _pfp.prism_fz(x2d, y2d, z1d)
            + _pfp.prism_fz(x1d, y2d, z1d)
            + _pfp.prism_fz(x2d, y1d, z1d)
            - _pfp.prism_fz(x1d, y1d, z1d)
        )

        K_chunk = (G_np * term).astype(np.float64, copy=False)
        K[:, i0:i1] = torch.from_numpy(K_chunk).to(device=device, dtype=dtype)

    return K, centers_v

mu0 = 4 * np.pi * 1e-7

def _prepare_bc_masks(mesh, tol=1e-8):
    """
    Ajusta la máscara para las condiciones de frontera durante el modelado
    
    Parámetros
    mesh = recibe la grilla 3D del model.
    
    Retorna
    bc_mask = máscara de las condiciones de frontera,
    top_x_edges = la parte superior de los bordes en la dirección x, 
    top_y_edges = la parte superior de los bordes en la dirección y.
    """
    
    zmax = mesh.nodes_z.max()
    top_x_edges = np.abs(mesh.edges_x[:, 2] - zmax) < tol
    top_y_edges = np.abs(mesh.edges_y[:, 2] - zmax) < tol

    bc_mask = np.zeros(mesh.nE, dtype=bool)
    bc_mask[:mesh.nEx][top_x_edges] = True
    bc_mask[mesh.nEx : mesh.nEx + mesh.nEy][top_y_edges] = True
    return bc_mask, top_x_edges, top_y_edges

def _prepare_solver(A, bc_mask):
    """
    Resuelve el sistema de ecuaciones dentro del dominio de interés usando las condiciones de frontera
    
    Parámetros
    A= sistema de ecuaciones, 
    bc_mask= máscara con las condiciones de frontera.
    
    Retorna
    free = zona de aire,
    fixed = zona de tierra,
    factor = solución del sistema de ecuaciones,
    Aib = zona de aire y tierra.
    """
    
    free = ~bc_mask
    fixed = bc_mask
    Aii = A[free][:, free].tocsc()
    Aib = A[free][:, fixed]
    factor = make_solver(Aii)
    return free, fixed, factor, Aib


def _solve_secondary(rhs, solver_data):
    """
    Resuelve el sistema de ecuaciones para encontrar el campo eléctrico secundario
    
    Parámetros
    rhs = fuente del sistema de ecuaciones,
    solver_data = operador aplicado sobre las incógnitas.
    
    Retorna
    sol= campo eléctrico secundario.
    """
    
    free, fixed, factor, Aib = solver_data
    rhs_reduced = rhs[free]  # zero Dirichlet for secondary

    sol = np.zeros_like(rhs, dtype=complex)
    sol[fixed] = 0.0
    sol[free] = factor(rhs_reduced)
    return sol


def _layered_1d_efield(nodes_z, sigma_1d, omega, mu=mu0):
    """
    Campo electrico magnetotelurico 1D E(z) evaluado en los nodos de la malla
    vertical, para un modelo estratificado ``sigma_1d`` (una conductividad por
    celda en z, ordenadas de abajo hacia arriba).

    Convenio armonico e^{+i omega t} y eje z positivo hacia arriba, de modo que
    la ecuacion de difusion es d2E/dz2 = i omega mu sigma E y la onda que decae
    hacia abajo es exp(+k z) con k = sqrt(i omega mu sigma), Re(k) > 0.

    En cada capa

        E(z)  = A e^{ k (z - z_j)} + B e^{-k (z - z_j)}
        H(z)  = -Y ( A e^{ k (z-z_j)} - B e^{-k (z-z_j)} ),   Y = k / (i omega mu)

    donde H es la componente horizontal ortogonal a E (par Ex/Hy). El semiespacio
    inferior solo admite la onda descendente (B = 0) y la propagacion se hace
    hacia arriba imponiendo continuidad de E y H en cada interfaz. El factor
    e^{+k h} se extrae en un acumulador logaritmico para no desbordar cuando el
    espesor de la capa supera varias veces el skin depth.

    El perfil se devuelve normalizado a E = 1 en el nodo superior de la malla.

    Parametros
    nodes_z    = coordenadas z de los nodos (nz + 1, crecientes),
    sigma_1d   = conductividad por celda (nz),
    omega      = frecuencia angular,
    mu         = permeabilidad magnetica.

    Retorna
    E = campo electrico en los nodos (nz + 1, complejo).
    """
    nodes_z = np.asarray(nodes_z, dtype=float)
    sigma_1d = np.asarray(sigma_1d, dtype=float)
    nz = sigma_1d.size
    if nodes_z.size != nz + 1:
        raise ValueError("nodes_z debe tener un elemento mas que sigma_1d")

    h = np.diff(nodes_z)
    k = np.sqrt(1j * omega * mu * sigma_1d)      # rama principal: Re(k) > 0
    Y = k / (1j * omega * mu)                    # admitancia intrinseca

    E_hat = np.zeros(nz + 1, dtype=complex)
    H_hat = np.zeros(nz + 1, dtype=complex)
    log_scale = np.zeros(nz + 1, dtype=complex)

    # Semiespacio inferior: solo onda descendente, amplitud unidad
    E_hat[0] = 1.0
    H_hat[0] = -Y[0]

    for j in range(nz):
        A = 0.5 * (E_hat[j] - H_hat[j] / Y[j])
        B = 0.5 * (E_hat[j] + H_hat[j] / Y[j])
        decay = np.exp(-2.0 * k[j] * h[j])       # |decay| <= 1
        E_hat[j + 1] = A + B * decay
        H_hat[j + 1] = -Y[j] * (A - B * decay)
        log_scale[j + 1] = log_scale[j] + k[j] * h[j]

    # Normalizacion al nodo superior. Re(log_scale - log_scale[-1]) <= 0, asi que
    # la exponencial esta acotada.
    return E_hat * np.exp(log_scale - log_scale[-1]) / E_hat[-1]


def _background_1d(mesh, sigma, sigma_primary=None):
    """
    Construye el modelo de fondo (primario) estratificado sobre el que se calcula
    el campo primario.

    El metodo primario/secundario exige que el campo primario resuelva de forma
    EXACTA el modelo de fondo. Por eso el fondo debe ser 1D (solo funcion de z) y
    debe usarse el MISMO modelo tanto para el campo primario como para la fuente
    ``-i omega M_{sigma - sigma_p} e_p``. Esta funcion devuelve las dos versiones
    consistentes entre si.

    Parametros
    mesh          = malla 3D,
    sigma         = modelo de conductividad verdadero (nC),
    sigma_primary = fondo. ``None`` -> semiespacio con el valor mas frecuente
                    bajo la superficie, conservando las celdas de aire; escalar
                    -> ese semiespacio, conservando las celdas de aire; arreglo
                    (nC) -> se usa tal cual.

    Retorna
    sigma_1d = conductividad por capa en z (nCz),
    sigma_p  = el mismo fondo expandido de vuelta a las nC celdas.
    """
    sigma = np.asarray(sigma, dtype=float)

    # El aire se identifica por su conductividad, no por z > 0: los modelos con
    # topografia tienen celdas de aire por debajo de z = 0 y marcarlas como
    # terreno mete un contraste espurio en el fondo.
    sigma_air = float(sigma.min())
    air = sigma <= sigma_air * (1.0 + 1e-6)

    if sigma_primary is None or np.isscalar(sigma_primary):
        if sigma_primary is None:
            below = sigma[~air]
            valores, cuentas = np.unique(below, return_counts=True)
            sigma_halfspace = float(valores[np.argmax(cuentas)])
        else:
            sigma_halfspace = float(sigma_primary)
        sigma_p3d = np.where(air, sigma_air, sigma_halfspace)
    else:
        sigma_p3d = np.asarray(sigma_primary, dtype=float)
        if sigma_p3d.size != mesh.nC:
            raise ValueError("sigma_primary debe ser escalar o de tamano mesh.nC")

    # Promedio en volumen de log(sigma) sobre la malla vertical, y vuelta a 3D:
    # el campo primario resuelve exactamente ese modelo 1D, no el 3D de partida.
    mesh1d = discretize.TensorMesh([mesh.h[-1]], [mesh.x0[-1]])
    mesh_col = discretize.TensorMesh(
        [[mesh.nodes_x[-1] - mesh.x0[0]], [mesh.nodes_y[-1] - mesh.x0[1]], mesh.h[-1]],
        x0=mesh.x0,
    )
    sigma_1d = np.exp(volume_average(mesh, mesh_col, np.log(sigma_p3d)))
    sigma_p = np.exp(volume_average(mesh_col, mesh, np.log(sigma_1d)))
    return sigma_1d, sigma_p


def _primary_fields(mesh, omega, sigma_1d, mu=mu0):
    """
    Calcula el campo electrico primario de onda plana para las polarizaciones x e y
    sobre el modelo de fondo estratificado ``sigma_1d``.

    Dos diferencias con la version anterior (``exp(k z)`` con un unico numero de
    onda para todo el dominio):

    1. El aire se trata con su propia conductividad. Antes el campo primario
       crecia como e^{z/delta} por encima de la superficie -- a 100 Hz y 5 km de
       aire eso son cuatro ordenes de magnitud de campo inventado -- cuando el
       campo MT en el aire es practicamente uniforme.

    2. El perfil se obtiene resolviendo el problema 1D DISCRETO, no el analitico.
       La descomposicion primario/secundario

           A_h(sigma) e_s = -i omega M_{sigma - sigma_p} e_p ,   e = e_p + e_s

       solo es exacta si ``A_h(sigma_p) e_p = 0`` en el sentido DISCRETO. Un
       perfil analitico satisface la ecuacion continua pero deja un residuo
       O(h^2) en la malla, y ese residuo se filtra al campo total. El perfil
       analitico se sigue usando, pero solo para fijar los valores de frontera
       (nodo superior e inferior) del sistema 1D.

    Parametros
    mesh     = malla 3D del modelo,
    omega    = frecuencia angular,
    sigma_1d = conductividad del fondo por capa en z (nCz), de abajo hacia arriba,
    mu       = permeabilidad magnetica.

    Retorna
    ex_pol = campo primario polarizado en x (nE),
    ey_pol = campo primario polarizado en y (nE).
    """
    mesh_1d = discretize.TensorMesh([mesh.h[-1]], [mesh.x0[-1]])

    # Operador 1D analogo a C^T M_{1/mu} C + i omega M_sigma
    G = mesh_1d.nodal_gradient
    M_mu = sp.diags(mesh_1d.cell_volumes * (1.0 / mu))
    M_sigma = mesh_1d.get_face_inner_product(sigma_1d)
    A_1d = (G.T @ M_mu @ G + 1j * omega * M_sigma).tocsc()

    # Frontera (nodo inferior y superior) tomada de la solucion analitica
    analitico = _layered_1d_efield(mesh_1d.nodes_x, sigma_1d, omega, mu)
    bc = np.r_[analitico[0], analitico[-1]]

    # A_ii e_i + A_io e_b = 0  ->  e_i = -A_ii^{-1} A_io e_b
    A_ii = A_1d[1:-1, 1:-1].tocsc()
    A_io = A_1d[1:-1][:, [0, -1]]
    e_interior = spla.spsolve(A_ii, -(A_io @ bc))

    E_nodes = np.r_[bc[0], e_interior, bc[1]]

    # Las aristas x e y de una TensorMesh se situan en coordenadas z nodales
    idx_x = np.searchsorted(mesh.nodes_z, mesh.edges_x[:, 2])
    idx_y = np.searchsorted(mesh.nodes_z, mesh.edges_y[:, 2])

    ex_pol = np.r_[
        E_nodes[idx_x],
        np.zeros(mesh.nEy, dtype=complex),
        np.zeros(mesh.nEz, dtype=complex),
    ]
    ey_pol = np.r_[
        np.zeros(mesh.nEx, dtype=complex),
        E_nodes[idx_y],
        np.zeros(mesh.nEz, dtype=complex),
    ]
    return ex_pol, ey_pol

    
def apparent_resistivity(Zcomp, frequencies):
    """
    Calcula la resistividad aparente a partir del tensor de impedancias usando las frecuencias solicitadas
    
    Parámetros
    Zcomp = Tensor de impedancias, 
    frequencies = Listado de frecuencias.
    
    Retorna
    Resistividad aparente
    """

    omega = 2 * np.pi * np.asarray(frequencies)[None, :]
    return np.abs(Zcomp) ** 2 / (mu0 * omega)

def phase_deg(Zcomp):
    """
    Calcula la fase aparente a partir del tensor de impedancias
    
    Parámetros
    Zcomp = Tensor de impedancias, 
    
    Retorna
    fase aparente
    """

    return np.degrees(np.angle(Zcomp))
    
def magnitude_field(Zcomp):
    """
    Calcula la magnitud del campo de entrada
    
    Parámetros
    Zcomp = Campo de entrada, 
    
    Retorna
    magnitud
    """

    return np.abs(Zcomp)
    
def compute_sigma_gradient(
    mesh,
    sigma,
    frequencies,
    Efull,
    Lambda
):
    """
    Calcula el gradiente del campo eléctrico en la dirección de sigma
    
    Parámetros
    mesh = recibe la grilla 3D del modelo,
    sigma= recibe el valor de conductividad actual,
    frequencies = recibe las frequencias de las mediciones,
    Efull = recibe el campo eléctrico total,
    Lambda = recibe el campo adjunto total
    
    Retorna
    grad = gradiente de la función de costo en la dirección de sigma
    """
    grad = np.zeros(mesh.nC)

    MeSigmaDeriv = mesh.get_edge_inner_product_deriv(
        sigma
    )

    for ifreq, freq in enumerate(frequencies):

        omega = 2*np.pi*freq

        for ipol in range(2):

            E = Efull[ifreq, ipol, :]
            Lam = Lambda[ifreq, ipol, :]

            grad += np.real(
                -1j * omega *
                (
                    MeSigmaDeriv(E)
                    .T.conjugate()
                    @ Lam
                )
            )

    return grad


def compute_sigma_gradient_Z(
    mesh,
    sigma,
    frequencies,
    Efull,
    Lambda
):
    """
    Calcula el gradiente de la funcion de costo basada en el tensor de impedancias en la dirección de la conductividad sigma
    
    Parámetros
    mesh = recibe la grilla 3D del modelo,
    sigma= recibe el valor de conductividad actual,
    frequencies = recibe las frequencias de las mediciones,
    Efull = recibe el campo eléctrico total,
    Lambda = recibe el campo adjunto total
    
    Retorna
    grad = gradiente de la función de costo en la dirección de sigma
    """
    grad = np.zeros(mesh.nC)

    MeSigmaDeriv = mesh.get_edge_inner_product_deriv(sigma)

    for ifreq, freq in enumerate(frequencies):

        omega = 2*np.pi*freq

        for ipol in range(2):

            E = Efull[ifreq, ipol, :]
            Lam = Lambda[ifreq, ipol, :]
            grad += np.real( 1j * omega * ( MeSigmaDeriv(E).T.conjugate() @ Lam ))

    return grad


def cost_and_gradient_log_Z(m, mesh, receivers, frequency_list, Z_obs,
                            sigma_primary=None):
    """
    Calcula el gradiente en la dirección del logaritmo de sigma y entrega su valor junto con la función de costo
    
    Parámetros
    m = recibe el valor de conductividad (sigma) actual,
    mesh = recibe la grilla 3D del modelo,
    receivers = recibe la ubicación de los receptores,
    frequency_list = recibe las frequencias en las que se hicieron las mediciones,
    Z_obs = recibe las mediciones del tensor de impedancias tomadas en campo.
    
    Retorna
    grad_m = gradiente de la función de costo en la dirección del logaritmo de sigma
    phi = el valor actual de la función de costo
    """
    
    import sys, os
    sys.path.append(os.path.abspath("../"))
    from Forward.MT3DZ import compute_mt_responses
    from Backward.Adjoint_sources import compute_adjoint_sources_Z
    from Backward.MTAdjoint import compute_mt_L_fields
    sigma = np.exp(m)

    """
    Modelado hacia adelante para obtener los valores del tensor de impedancias usando la conductividad actual
    """
    
    fields, responses = compute_mt_responses(
        mesh,
        sigma,
        receivers,
        frequency_list,
        sigma_primary=sigma_primary
    )

    """
    Cálculo de los residuales
    """
    
    residual = responses.Z - Z_obs

    """
    Cálculo de la función de costo para el valor actual de conductividad
    """
    
    phi = 0.5*np.vdot(
        residual,
        residual
    ).real

    """
    Cálculo de las fuentes adjuntas
    """
    adj_sources = compute_adjoint_sources_Z(
        mesh,
        receivers,
        frequency_list,
        fields,
        responses,
        Z_obs
    )
    
    """
    Cálculo del campo adjunto al campo eléctrico
    """
    
    MTAdjointFields = compute_mt_L_fields(
        mesh,
        sigma,
        frequency_list,
        adj_sources.rhs_x,
        adj_sources.rhs_y
    )

    """
    Cálculo del gradiente
    """
    
    grad_sigma = compute_sigma_gradient_Z(
        mesh,
        sigma,
        frequency_list,
        fields.Efull,
        MTAdjointFields.Lambda
    )

    grad_m = sigma * grad_sigma

    return phi, grad_m

def cost_and_gradient_log(m, mesh, receivers, frequency_list, E_obs):
    """
    Calcula el gradiente en la dirección del logaritmo de sigma y entrega su valor junto con la función de costo
    
    Parámetros
    m = recibe el valor de conductividad (sigma) actual,
    mesh = recibe la grilla 3D del modelo,
    receivers = recibe la ubicación de los receptores,
    frequency_list = recibe las frequencias en las que se hicieron las mediciones,
    E_obs = recibe las mediciones del campo eléctrico tomadas en campo.
    
    Retorna
    grad_m = gradiente de la función de costo en la dirección del logaritmo de sigma
    phi = el valor actual de la función de costo
    """
    
    import sys, os
    sys.path.append(os.path.abspath("../"))
    from Forward.MT3DE import compute_mt_E_fields
    from Backward.MT3DL2 import compute_mt_L_fields
    sigma = np.exp(m)

    """
    Modelado hacia adelante para obtener los valores del campo eléctrico usando la conductividad actual
    """
    Ereceiver, Efull = compute_mt_E_fields(
        mesh,
        sigma,
        receivers,
        frequency_list
    )

    """
    Cálculo de los residuales
    """

    residual = Ereceiver - E_obs

    """
    Cálculo de la función de costo para el valor actual de conductividad
    """
    
    phi = 0.5*np.vdot(
        residual,
        residual
    ).real

    """
    Cálculo del campo adjunto al campo eléctrico
    """
    
    Lambda = compute_mt_L_fields(
        mesh,
        sigma,
        receivers,
        frequency_list,
        residual[:,:,0],
        residual[:,:,1]
    )
    
    """
    Cálculo del gradiente
    """

    grad_sigma = compute_sigma_gradient(
        mesh,
        sigma,
        frequency_list,
        Efull,
        Lambda
    )

    grad_m = sigma * grad_sigma

    return phi, grad_m

def cost_and_gradient(sigma, mesh, receivers, frequency_list, E_obs):
    """
    Calcula el gradiente del campo eléctrico en la dirección de sigma y entrega su valor junto con la función de costo
    
    Parámetros
    sigma = recibe el valor de conductividad actual,
    mesh = recibe la grilla 3D del modelo,
    receivers = recibe la ubicación de los receptores,
    frequency_list = recibe las frequencias en las que se hicieron las mediciones,
    E_obs = recibe las mediciones del campo eléctrico tomadas en campo.
    
    Retorna
    grad = gradiente de la función de costo en la dirección del logaritmo de sigma
    phi = el valor actual de la función de costo
    """
    
    import sys, os
    sys.path.append(os.path.abspath("../"))
    from Forward.MT3DE import compute_mt_E_fields
    from Backward.MT3DL2 import compute_mt_L_fields
    
    """
    Modelado hacia adelante para obtener los valores del campo eléctrico usando la conductividad actual
    """
    
    Ereceiver, Efull = compute_mt_E_fields(
        mesh,
        sigma,
        receivers,
        frequency_list
    )

    """
    Cálculo de los residuales
    """
    
    residual = Ereceiver - E_obs

    """
    Cálculo de la función de costo para el valor actual de conductividad
    """
    
    phi = 0.5 * np.vdot(residual, residual).real

    """
    Cálculo del campo adjunto al campo eléctrico
    """
    
    Lambda = compute_mt_L_fields(
        mesh,
        sigma,
        receivers,
        frequency_list,
        residual[:,:,0],
        residual[:,:,1]
    )

    """
    Cálculo del gradiente
    """
    
    grad = compute_sigma_gradient(
        mesh,
        sigma,
        frequency_list,
        Efull,
        Lambda
    )

    return phi, grad

def cost_and_gradient_Z(sigma, mesh, receivers, frequency_list, Z_obs,
                        sigma_primary=None):
    """
    Calcula el gradiente del tensor de impedancias en la dirección de sigma y entrega su valor junto con la función de costo
    
    Parámetros
    sigma = recibe el valor de conductividad actual,
    mesh = recibe la grilla 3D del modelo,
    receivers = recibe la ubicación de los receptores,
    frequency_list = recibe las frequencias en las que se hicieron las mediciones,
    Z_obs = recibe las mediciones del tensor de impedancias tomadas en campo.
    
    Retorna
    grad = gradiente de la función de costo en la dirección del logaritmo de sigma
    phi = el valor actual de la función de costo
    """
    
    import sys, os
    sys.path.append(os.path.abspath("../"))
    from Forward.MT3DZ import compute_mt_responses
    from Backward.Adjoint_sources import compute_adjoint_sources_Z
    from Backward.MTAdjoint import compute_mt_L_fields
    
    """
    Modelado hacia adelante para obtener los valores del tensor de impedancias usando la conductividad actual
    """
    
    fields, responses = compute_mt_responses(
        mesh,
        sigma,
        receivers,
        frequency_list,
        sigma_primary=sigma_primary
    )

    """
    Cálculo de los residuales
    """
    
    residual = responses.Z - Z_obs

    """
    Cálculo de la función de costo para el valor actual de conductividad
    """
    
    phi = 0.5 * np.vdot(residual, residual).real

    """
    Cálculo de las fuentes adjuntas
    """
    adj_sources = compute_adjoint_sources_Z(
        mesh,
        receivers,
        frequency_list,
        fields,
        responses,
        Z_obs
    )
    
    """
    Cálculo del campo adjunto al campo eléctrico
    """
    
    MTAdjointFields = compute_mt_L_fields(
        mesh,
        sigma,
        frequency_list,
        adj_sources.rhs_x,
        adj_sources.rhs_y
    )

    """
    Cálculo del gradiente
    """
    
    grad = compute_sigma_gradient_Z(
        mesh,
        sigma,
        frequency_list,
        fields.Efull,
        MTAdjointFields.Lambda
    )

    return phi, grad

def line_search_log(
    m,
    grad,
    phi0,
    active,
    mesh, 
    receivers, 
    frequency_list, 
    E_obs,
    alpha0=1.0,
    c=1e-4,
    tau=0.5,
    max_iter=20
):
    """
    Esta función minimiza la función de costo usando el método del máximo descenso. Para ello necesita la información del gradiente junto con un valor inicial de conductividad
    Parámetros
    m = recibe el valor de conductividad (sigma) actual,
    grad = recibe el valor del gradiente en la posición actual,
    phi0 = recibe el valor de la función de costo en la posición actual,
    active = recibe las celdas activas del modelo,
    mesh = recibe la grilla 3D del modelo,
    receivers = recibe la ubicación de los receptores,
    frequency_list = recibe las frequencias en las que se hicieron las mediciones,
    E_obs = recibe las mediciones del campo eléctrico tomadas en campo.
    alpha0 = Valor de avance inicial en la dirección de menos el gradiente,
    c = factor de escala aplicado sobre alpha para aceptar un nuevo punto de avance,
    tau = factor de escala aplicado sobre alpha en cada intento,
    max_iter = número de intentos permitidos para encontrar un mejor punto
    
    Retorna
    alpha = valor de avance con el que la función de costo decrece, 
    phi_trial = nuevo valor de la función de costo
    """
    
    alpha = alpha0

    g2 = np.dot(
        grad[active],
        grad[active]
    )

    for _ in range(max_iter):

        m_trial = m.copy()

        m_trial[active] -= alpha*grad[active]

        phi_trial, _ = cost_and_gradient_log(
            m_trial, mesh, receivers, frequency_list, E_obs
        )

        if phi_trial <= phi0 - c*alpha*g2:
            return alpha, phi_trial

        alpha *= tau

    return alpha, phi_trial
    
def line_search(
    sigma,
    grad,
    phi0,
    active,
    mesh, 
    receivers, 
    frequency_list, 
    E_obs,
    alpha0=1.0,
    c=1e-4,
    tau=0.5,
    max_iter=20
):
    """
    Esta función minimiza la función de costo usando el método del máximo descenso. Para ello necesita la información del gradiente junto con un valor inicial de conductividad
    Parámetros
    sigma = recibe el valor de conductividad actual,
    grad = recibe el valor del gradiente en la posición actual,
    phi0 = recibe el valor de la función de costo en la posición actual,
    active = recibe las celdas activas del modelo,
    mesh = recibe la grilla 3D del modelo,
    receivers = recibe la ubicación de los receptores,
    frequency_list = recibe las frequencias en las que se hicieron las mediciones,
    E_obs = recibe las mediciones del campo eléctrico tomadas en campo.
    alpha0 = Valor de avance inicial en la dirección de menos el gradiente,
    c = factor de escala aplicado sobre alpha para aceptar un nuevo punto de avance,
    tau = factor de escala aplicado sobre alpha en cada intento,
    max_iter = número de intentos permitidos para encontrar un mejor punto
    
    Retorna
    alpha = valor de avance con el que la función de costo decrece, 
    phi_trial = nuevo valor de la función de costo
    """
    
    alpha = alpha0

    g2 = np.dot(
        grad[active],
        grad[active]
    )

    for _ in range(max_iter):

        sigma_trial = sigma.copy()

        sigma_trial[active] -= alpha * grad[active]

        sigma_trial[active] = np.maximum(
            sigma_trial[active],
            1e-8
        )

        phi_trial, _ = cost_and_gradient(
            sigma_trial, mesh, receivers, frequency_list, E_obs
        )

        if phi_trial <= phi0 - c*alpha*g2:
            return alpha, phi_trial

        alpha *= tau

    return alpha, phi_trial
    
def line_search_Z(
    sigma,
    grad,
    phi0,
    active,
    mesh, 
    receivers, 
    frequency_list, 
    Z_obs,
    alpha0=1.0,
    c=1e-4,
    tau=0.5,
    max_iter=20
):
    """
    Esta función minimiza la función de costo usando el método del máximo descenso. Para ello necesita la información del gradiente junto con un valor inicial de conductividad
    Parámetros
    sigma = recibe el valor de conductividad actual,
    grad = recibe el valor del gradiente en la posición actual,
    phi0 = recibe el valor de la función de costo en la posición actual,
    active = recibe las celdas activas del modelo,
    mesh = recibe la grilla 3D del modelo,
    receivers = recibe la ubicación de los receptores,
    frequency_list = recibe las frequencias en las que se hicieron las mediciones,
    Z_obs = recibe las mediciones del tensor de impedancias tomadas en campo.
    alpha0 = Valor de avance inicial en la dirección de menos el gradiente,
    c = factor de escala aplicado sobre alpha para aceptar un nuevo punto de avance,
    tau = factor de escala aplicado sobre alpha en cada intento,
    max_iter = número de intentos permitidos para encontrar un mejor punto
    
    Retorna
    alpha = valor de avance con el que la función de costo decrece, 
    phi_trial = nuevo valor de la función de costo
    """
    
    alpha = alpha0

    g2 = np.dot(
        grad[active],
        grad[active]
    )

    for _ in range(max_iter):

        sigma_trial = sigma.copy()

        sigma_trial[active] -= alpha * grad[active]

        sigma_trial[active] = np.maximum(
            sigma_trial[active],
            1e-8
        )

        phi_trial, _ = cost_and_gradient_Z(
            sigma_trial, mesh, receivers, frequency_list, Z_obs
        )

        if phi_trial <= phi0 - c*alpha*g2:
            return alpha, phi_trial

        alpha *= tau

    return alpha, phi_trial
    
def line_search_log_Z(
    m,
    grad,
    phi0,
    active,
    mesh, 
    receivers, 
    frequency_list, 
    Z_obs,
    alpha0=1.0,
    c=1e-4,
    tau=0.5,
    max_iter=20
):
    """
    Esta función minimiza la función de costo usando el método del máximo descenso. Para ello necesita la información del gradiente junto con un valor inicial de conductividad
    Parámetros
    m = recibe el valor de conductividad (sigma) actual,
    grad = recibe el valor del gradiente en la posición actual,
    phi0 = recibe el valor de la función de costo en la posición actual,
    active = recibe las celdas activas del modelo,
    mesh = recibe la grilla 3D del modelo,
    receivers = recibe la ubicación de los receptores,
    frequency_list = recibe las frequencias en las que se hicieron las mediciones,
    Z_obs = recibe las mediciones del tensor de impedancias tomadas en campo.
    alpha0 = Valor de avance inicial en la dirección de menos el gradiente,
    c = factor de escala aplicado sobre alpha para aceptar un nuevo punto de avance,
    tau = factor de escala aplicado sobre alpha en cada intento,
    max_iter = número de intentos permitidos para encontrar un mejor punto
    
    Retorna
    alpha = valor de avance con el que la función de costo decrece, 
    phi_trial = nuevo valor de la función de costo
    """
    
    alpha = alpha0

    g2 = np.dot(
        grad[active],
        grad[active]
    )

    for _ in range(max_iter):

        m_trial = m.copy()

        m_trial[active] -= alpha*grad[active]

        phi_trial, _ = cost_and_gradient_log_Z(
            m_trial, mesh, receivers, frequency_list, Z_obs
        )

        if phi_trial <= phi0 - c*alpha*g2:
            return alpha, phi_trial

        alpha *= tau

    return alpha, phi_trial
