import numpy as np
import sys, os
sys.path.append(os.path.abspath("../"))
from Forward.utils import _prepare_bc_masks
from Forward.utils import _primary_fields
from Forward.utils import _prepare_solver
from Forward.utils import _solve_secondary

mu0 = 4 * np.pi * 1e-7

def compute_mt_impedance_tipper(mesh, sigma, receivers, frequencies, mu=None):
    """
    Calcula el tensor de impedancia Z y el tipper T para múltiples frecuencias y receptores
    
    Parámetros
    
    mesh: Grilla 3D del modelo
    sigma: Modelo de conductividad 
    receivers: Ubicación de la(s) antena(s)
    frequencies: Conjunto de frecuencias recibidas por la(s) antena(s)

    Retorna
    -------
    Z : array (n_rec, n_freq, 2, 2), Tensor de impedancias
    T : array (n_rec, n_freq, 2) con componentes [Tzx, Tzy]
    """
    if mu is None:
        mu = mu0 * np.ones(mesh.nC)

    receivers = np.atleast_2d(receivers)
    n_rec = receivers.shape[0]
    n_freq = len(frequencies)
    Z = np.zeros((n_rec, n_freq, 2, 2), dtype=complex)
    T = np.zeros((n_rec, n_freq, 2), dtype=complex)

    C = mesh.edge_curl
    bc_mask, _, _ = _prepare_bc_masks(mesh)

    below_surface = mesh.cell_centers[:, 2] <= 0
    sigma_background = 1e-2

    mu_faces = mesh.aveCC2F @ mu

    nFx, nFy = mesh.nFx, mesh.nFy

    Pex = mesh.get_interpolation_matrix(receivers,"Ex")

    Pey = mesh.get_interpolation_matrix(receivers,"Ey")

    Pfy = mesh.get_interpolation_matrix(receivers,"Fy")

    Pfx = mesh.get_interpolation_matrix(receivers,"Fx")

    Pfz = mesh.get_interpolation_matrix(receivers,"Fz")

    for ifreq, freq in enumerate(frequencies):
        print(f'Resolviendo para la frecuencia {ifreq} de {len(frequencies)}')

        omega = 2 * np.pi * freq
        Msigma = mesh.get_edge_inner_product(sigma)
        Mmu_inv = mesh.get_face_inner_product(1 / mu)
        A = C.T @ Mmu_inv @ C + 1j * omega * Msigma

        primary_ex, primary_ey = _primary_fields(mesh, omega, sigma_background)
        solver_data = _prepare_solver(A, bc_mask)

        # Lado derecho del sistema (RHS)
        delta_sigma = np.zeros(mesh.nC)
        delta_sigma[below_surface] = sigma[below_surface] - sigma_background # simpeg trabaja con la diferencia de sigma
        Mdelta_sigma = mesh.get_edge_inner_product(delta_sigma)
        
        # Polarización Ex
        rhs_ex = -1j * omega * (Mdelta_sigma @ primary_ex) # fuente de simpeg
        ex_total = primary_ex + _solve_secondary(rhs_ex, solver_data)

        # Polarización Ey
        rhs_ey = -1j * omega * (Mdelta_sigma @ primary_ey) # fuente de simpeg
        ey_total = primary_ey + _solve_secondary(rhs_ey, solver_data)

        # Campos magnéticos a partir de la ley de Faraday
        curlEx = C @ ex_total
        curlEy = C @ ey_total

        B_from_Ex = -(1 / (1j * omega)) * curlEx
        B_from_Ey = -(1 / (1j * omega)) * curlEy

        H_from_Ex = B_from_Ex / mu_faces
        H_from_Ey = B_from_Ey / mu_faces

        Hx_from_Ex = H_from_Ex[:nFx]
        Hy_from_Ex = H_from_Ex[nFx : nFx + nFy]
        Hz_from_Ex = H_from_Ex[nFx + nFy :]

        Hx_from_Ey = H_from_Ey[:nFx]
        Hy_from_Ey = H_from_Ey[nFx : nFx + nFy]
        Hz_from_Ey = H_from_Ey[nFx + nFy :]

        for ir, rx in enumerate(receivers):
            print(f'calculando la impedancia en el receptor {ir:3d}/{len(receivers)} at {rx}, por favor espere :)')
            
            Ex_val = (Pex @ ex_total)[ir]
            Ey_val = (Pey @ ey_total)[ir]

            Hx_val_ex = (Pfx @ H_from_Ex)[ir]
            Hy_val_ex = (Pfy @ H_from_Ex)[ir]
            Hz_val_ex = (Pfz @ H_from_Ex)[ir]
            
            Hx_val_ey = (Pfx @ H_from_Ey)[ir]
            Hy_val_ey = (Pfy @ H_from_Ey)[ir]
            Hz_val_ey = (Pfz @ H_from_Ey)[ir]
            
            Zxy = Ex_val / Hy_val_ex
            Zyx = Ey_val / Hx_val_ey
            Zxx = Ex_val / Hx_val_ex
            Zyy = Ey_val / Hy_val_ey

            Z[ir, ifreq, 0, 0] = Zxx
            Z[ir, ifreq, 0, 1] = Zxy
            Z[ir, ifreq, 1, 0] = Zyx
            Z[ir, ifreq, 1, 1] = Zyy

            A_mat = np.array( [ [Hx_val_ex, Hy_val_ex], [Hx_val_ey, Hy_val_ey] ], dtype=complex )
            b_vec = np.array([Hz_val_ex, Hz_val_ey], dtype=complex)
            
            try:
                T[ir, ifreq, :] = np.linalg.solve(A_mat, b_vec)
            except np.linalg.LinAlgError:
                T[ir, ifreq, :] = np.nan

    return Z, T
