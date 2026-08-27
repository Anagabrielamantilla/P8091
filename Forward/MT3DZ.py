import numpy as np
from dataclasses import dataclass
import sys, os
sys.path.append(os.path.abspath("../"))
from Forward.utils import _prepare_bc_masks
from Forward.utils import _primary_fields
from Forward.utils import _background_1d
from Forward.utils import _prepare_solver
from Forward.utils import _solve_secondary

@dataclass
class MTFields:
    Ereceiver: np.ndarray
    Hreceiver: np.ndarray
    Efull: np.ndarray
    Hfull: np.ndarray

@dataclass
class MTResponses:
    Z: np.ndarray
    T: np.ndarray

mu0 = 4 * np.pi * 1e-7

def compute_mt_responses(mesh, sigma, receivers, frequencies, mu=None,
                         sigma_primary=None, boundary_condition="natural"):
    """
    Calcula el campo electrico y el campo magnetico en las polarizaciones x e y para
    multiples frecuencias y receptores. Tambien calcula el tensor de impedancia Z y el
    tipper T para multiples frecuencias y receptores.

    Parametros

    mesh: Grilla 3D del modelo
    sigma: Modelo de conductividad
    receivers: Ubicacion de la(s) antena(s)
    frequencies: Conjunto de frecuencias recibidas por la(s) antena(s)
    mu: Permeabilidad magnetica por celda. Por defecto mu0.
    sigma_primary: Modelo de fondo para el campo primario. None -> semiespacio con la
        conductividad mas frecuente bajo la superficie, conservando el aire; escalar ->
        ese semiespacio; arreglo (nC) -> se usa tal cual. El campo primario y la fuente
        secundaria SIEMPRE usan el mismo fondo.
    boundary_condition: condicion de frontera del campo SECUNDARIO.
        "natural"       -> n x (mu^-1 curl e_s) = 0 en todo el borde, que es la
                           condicion que implica la discretizacion de Galerkin
                           cuando no se elimina ninguna fila. Es la opcion por
                           defecto y la que usa SimPEG.
        "dirichlet_top" -> ademas e_s = 0 en las aristas horizontales del techo.
                           Comportamiento historico del operador. Forzar a cero
                           el campo secundario a 5 km de altura mientras los
                           lados quedan libres es inconsistente y contamina sobre
                           todo el tipper a baja frecuencia.

    Retorna
    -------
    responses:
    Z : array (n_rec, n_freq, 2, 2), Tensor de impedancias
    T : array (n_rec, n_freq, 2) con componentes [Tzx, Tzy]
    fields:
    Ereceiver : array (n_rec, n_freq, 2, 2) campo electrico en el receptor,
                indices [receptor, frecuencia, componente (x,y), polarizacion (x,y)]
    Efull : array (n_freq, 2, mesh.nE) campo electrico para todas las frecuencias y polarizaciones
    Hreceiver : array (n_rec, n_freq, 3, 2) campo magnetico en el receptor,
                indices [receptor, frecuencia, componente (x,y,z), polarizacion (x,y)]
    Hfull : array (n_freq, 2, mesh.nF) campo magnetico para todas las frecuencias y polarizaciones
    """
    if mu is None:
        mu = mu0 * np.ones(mesh.nC)

    receivers = np.atleast_2d(receivers)
    n_rec = receivers.shape[0]
    n_freq = len(frequencies)

    Ereceiver = np.zeros((n_rec, n_freq, 2, 2), dtype=complex)
    Efull = np.zeros((n_freq, 2, mesh.nE), dtype=complex)

    Hreceiver = np.zeros((n_rec, n_freq, 3, 2), dtype=complex)
    Hfull = np.zeros((n_freq, 2, mesh.nF), dtype=complex)

    Z = np.zeros((n_rec, n_freq, 2, 2), dtype=complex)
    T = np.zeros((n_rec, n_freq, 2), dtype=complex)

    C = mesh.edge_curl
    if boundary_condition == "natural":
        bc_mask = np.zeros(mesh.nE, dtype=bool)
    elif boundary_condition == "dirichlet_top":
        bc_mask, _, _ = _prepare_bc_masks(mesh)
    else:
        raise ValueError(
            f"boundary_condition desconocida: {boundary_condition!r}. "
            "Use 'natural' o 'dirichlet_top'."
        )

    mu_faces = mesh.aveCC2F @ mu

    Pex = mesh.get_interpolation_matrix(receivers,"Ex")

    Pey = mesh.get_interpolation_matrix(receivers,"Ey")

    Pfx = mesh.get_interpolation_matrix(receivers,"Fx")

    Pfy = mesh.get_interpolation_matrix(receivers,"Fy")

    Pfz = mesh.get_interpolation_matrix(receivers,"Fz")

    Msigma = mesh.get_edge_inner_product(sigma)
    Mmu_inv = mesh.get_face_inner_product(1 / mu)

    # Fondo estratificado. sigma_1d alimenta el campo primario y sigma_p la fuente
    # secundaria: los dos tienen que salir del MISMO modelo para que
    # A(sigma_p) e_p = 0 y la descomposicion primario/secundario sea exacta.
    sigma_1d, sigma_p = _background_1d(mesh, sigma, sigma_primary)

    # Lado derecho del sistema (RHS)
    delta_sigma = np.asarray(sigma, dtype=float) - sigma_p
    Mdelta_sigma = mesh.get_edge_inner_product(delta_sigma)

    for ifreq, freq in enumerate(frequencies):
        print(f'Resolviendo para la frecuencia {ifreq} de {len(frequencies)}')

        omega = 2 * np.pi * freq
        A = C.T @ Mmu_inv @ C + 1j * omega * Msigma

        primary_ex, primary_ey = _primary_fields(mesh, omega, sigma_1d)
        solver_data = _prepare_solver(A, bc_mask)

        # Polarizacion Ex
        rhs_ex = -1j * omega * (Mdelta_sigma @ primary_ex)
        ex_total = primary_ex + _solve_secondary(rhs_ex, solver_data)

        # Polarizacion Ey
        rhs_ey = -1j * omega * (Mdelta_sigma @ primary_ey)
        ey_total = primary_ey + _solve_secondary(rhs_ey, solver_data)

        # Campos magneticos a partir de la ley de Faraday
        curlEx = C @ ex_total
        curlEy = C @ ey_total

        B_from_Ex = -(1 / (1j * omega)) * curlEx
        B_from_Ey = -(1 / (1j * omega)) * curlEy

        H_from_Ex = B_from_Ex / mu_faces
        H_from_Ey = B_from_Ey / mu_faces

        Efull[ifreq,0,:] = ex_total
        Efull[ifreq,1,:] = ey_total

        Hfull[ifreq,0,:]=H_from_Ex
        Hfull[ifreq,1,:]=H_from_Ey

        # Las DOS componentes horizontales de E en las DOS polarizaciones.
        # Ex_ey y Ey_ex son nulas solo en un medio 1D; en 3D son parte del tensor.
        Ex_ex = Pex @ ex_total
        Ey_ex = Pey @ ex_total

        Ex_ey = Pex @ ey_total
        Ey_ey = Pey @ ey_total

        Hx_ex = Pfx @ H_from_Ex
        Hy_ex = Pfy @ H_from_Ex
        Hz_ex = Pfz @ H_from_Ex

        Hx_ey = Pfx @ H_from_Ey
        Hy_ey = Pfy @ H_from_Ey
        Hz_ey = Pfz @ H_from_Ey

        for ir, rx in enumerate(receivers):
            print(f'calculando la impedancia en el receptor {ir:3d}/{len(receivers)} at {rx}, por favor espere :)')

            Ereceiver[ir, ifreq, 0, 0] = Ex_ex[ir]
            Ereceiver[ir, ifreq, 1, 0] = Ey_ex[ir]
            Ereceiver[ir, ifreq, 0, 1] = Ex_ey[ir]
            Ereceiver[ir, ifreq, 1, 1] = Ey_ey[ir]

            Hx_val_ex = Hx_ex[ir]
            Hy_val_ex = Hy_ex[ir]
            Hz_val_ex = Hz_ex[ir]

            Hx_val_ey = Hx_ey[ir]
            Hy_val_ey = Hy_ey[ir]
            Hz_val_ey = Hz_ey[ir]

            Hreceiver[ir, ifreq, 0, 0] = Hx_val_ex
            Hreceiver[ir, ifreq, 1, 0] = Hy_val_ex
            Hreceiver[ir, ifreq, 2, 0] = Hz_val_ex

            Hreceiver[ir, ifreq, 0, 1] = Hx_val_ey
            Hreceiver[ir, ifreq, 1, 1] = Hy_val_ey
            Hreceiver[ir, ifreq, 2, 1] = Hz_val_ey

            # E = Z H con las columnas indexadas por polarizacion:
            #   EH = [[Ex(x), Ex(y)], [Ey(x), Ey(y)]]
            #   HH = [[Hx(x), Hx(y)], [Hy(x), Hy(y)]]
            #   Z  = EH @ HH^{-1}   <=>   HH^T Z^T = EH^T
            EH = np.array([[Ex_ex[ir], Ex_ey[ir]],
                           [Ey_ex[ir], Ey_ey[ir]]], dtype=complex)

            HH = np.array([[Hx_ex[ir], Hx_ey[ir]],
                           [Hy_ex[ir], Hy_ey[ir]]], dtype=complex)

            cond = np.linalg.cond(HH)
            if cond < 1e12:
                Z[ir, ifreq] = np.linalg.solve(HH.T, EH.T).T
            else:
                Z[ir, ifreq] = np.nan

            # Hz = Tzx Hx + Tzy Hy en cada polarizacion
            A_mat = np.array( [ [Hx_val_ex, Hy_val_ex], [Hx_val_ey, Hy_val_ey] ], dtype=complex )
            b_vec = np.array([Hz_val_ex, Hz_val_ey], dtype=complex)

            cond = np.linalg.cond(A_mat)
            if cond < 1e12:
                T[ir, ifreq,:] = np.linalg.solve(A_mat,b_vec)
            else:
                T[ir,ifreq,:] = np.nan


    fields = MTFields(
        Ereceiver,
        Hreceiver,
        Efull,
        Hfull,
    )

    responses = MTResponses(
        Z,
        T,
    )

    return fields, responses
