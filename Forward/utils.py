import math
import numpy as np
import scipy.sparse.linalg as spla
import torch
from geoana.kernels import prism_fzz, prism_fzx, prism_fzy
from geoana.kernels import potential_field_prism as _pfp

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
    factor = spla.factorized(Aii)
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


def _primary_fields(mesh, omega, sigma_background):
    """
    Calcula el campo eléctrico primario para ondas planas polarizadas en la dirección x, y.
    
    Parámetros
    mesh = recibe la grilla 3D del modelo,
    omega = frecuencia angular, 
    sigma_background = conductividad de referencia.
    
    Retorna
    ex_pol() = campo eléctrico primario polarizado en dirección x, 
    ey_pol() = campo eléctrico primario polarizado en dirección y.
    """
    
    k_wave = (1 + 1j) / np.sqrt(2 / (omega * mu0 * sigma_background))

    def ex_pol():
        depth = mesh.edges_x[:, 2]
        ex_primary = np.exp(k_wave * depth)
        ey_primary = np.zeros(mesh.nEy, dtype=complex)
        ez_primary = np.zeros(mesh.nEz, dtype=complex)
        return np.r_[ex_primary, ey_primary, ez_primary]

    def ey_pol():
        depth = mesh.edges_y[:, 2]
        ex_primary = np.zeros(mesh.nEx, dtype=complex)
        ey_primary = np.exp(k_wave * depth)
        ez_primary = np.zeros(mesh.nEz, dtype=complex)
        return np.r_[ex_primary, ey_primary, ez_primary]

    return ex_pol(), ey_pol()

    
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
    Calcula el gradiente en la dirección de sigma
    
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
    Calcula el gradiente en la dirección de sigma y entrega su valor junto con la función de costo
    
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