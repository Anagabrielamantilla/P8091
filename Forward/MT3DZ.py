import numpy as np
from dataclasses import dataclass
import sys, os
sys.path.append(os.path.abspath("../"))
from Forward.utils import _prepare_bc_masks
from Forward.utils import _primary_fields
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

def compute_mt_responses(mesh, sigma, receivers, frequencies, mu=None):
    """
    Calcula el campo eléctrico, el campo magnético en las polarizaciones x y z para múltiples frecuencias y receptores. También calcula el tensor de impedancia Z y el tipper T para múltiples frecuencias y receptores
    
    Parámetros
    
    mesh: Grilla 3D del modelo
    sigma: Modelo de conductividad 
    receivers: Ubicación de la(s) antena(s)
    frequencies: Conjunto de frecuencias recibidas por la(s) antena(s)

    Retorna
    -------
    responses:
    Z : array (n_rec, n_freq, 2, 2), Tensor de impedancias
    T : array (n_rec, n_freq, 2) con componentes [Tzx, Tzy]
    fields:
    Ereceiver : array (n_rec, n_freq, 2) campo eléctrico para todos los receptores, las frecuencias y las polarizaciones
    Efull : array (n_freq, 2, mesh.nE) campo eléctrico para todas las frecuencias y polarizaciones
    Hreceiver : array (n_rec, n_freq, 3, 2) campo magnético para todos los receptores, las frecuencias y las polarizaciones
    Hfull : array (n_freq, 2, mesh.nF) campo magnético para todas las frecuencias y polarizaciones
    """
    if mu is None:
        mu = mu0 * np.ones(mesh.nC)

    receivers = np.atleast_2d(receivers)
    n_rec = receivers.shape[0]
    n_freq = len(frequencies)
    
    Ereceiver = np.zeros((n_rec, n_freq, 2), dtype=complex)
    Efull = np.zeros((n_freq, 2, mesh.nE), dtype=complex)
    
    Hreceiver = np.zeros((n_rec, n_freq, 3, 2), dtype=complex)
    Hfull = np.zeros((n_freq, 2, mesh.nF), dtype=complex)
    
    Z = np.zeros((n_rec, n_freq, 2, 2), dtype=complex)
    T = np.zeros((n_rec, n_freq, 2), dtype=complex)

    C = mesh.edge_curl
    bc_mask, _, _ = _prepare_bc_masks(mesh)

    below_surface = mesh.cell_centers[:, 2] <= 0
    sigma_background = 1e-2

    mu_faces = mesh.aveCC2F @ mu

    Pex = mesh.get_interpolation_matrix(receivers,"Ex")

    Pey = mesh.get_interpolation_matrix(receivers,"Ey")
    
    Pfx = mesh.get_interpolation_matrix(receivers,"Fx")

    Pfy = mesh.get_interpolation_matrix(receivers,"Fy")

    Pfz = mesh.get_interpolation_matrix(receivers,"Fz")
    
    Msigma = mesh.get_edge_inner_product(sigma)
    Mmu_inv = mesh.get_face_inner_product(1 / mu)

    # RHS
    delta_sigma = np.zeros(mesh.nC)
    delta_sigma[below_surface] = sigma[below_surface] - sigma_background # simpeg trabaja con la diferencia de sigma. Sergio
    Mdelta_sigma = mesh.get_edge_inner_product(delta_sigma)

    for ifreq, freq in enumerate(frequencies):
        print(f'Resolviendo para la frecuencia {ifreq} de {len(frequencies)}')

        omega = 2 * np.pi * freq
        A = C.T @ Mmu_inv @ C + 1j * omega * Msigma

        primary_ex, primary_ey = _primary_fields(mesh, omega, sigma_background)
        solver_data = _prepare_solver(A, bc_mask)
    
        # Ex polarization
        rhs_ex = -1j * omega * (Mdelta_sigma @ primary_ex) # fuente de simpeg. Sergio
        ex_total = primary_ex + _solve_secondary(rhs_ex, solver_data)

        # Ey polarization
        rhs_ey = -1j * omega * (Mdelta_sigma @ primary_ey) # fuente de simpeg. Sergio
        ey_total = primary_ey + _solve_secondary(rhs_ey, solver_data)

        # Magnetic fields from Faraday's law.
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
        
        Ex_obs = Pex @ ex_total
        Ey_obs = Pey @ ey_total
        
        Hx_ex = Pfx @ H_from_Ex
        Hy_ex = Pfy @ H_from_Ex
        Hz_ex = Pfz @ H_from_Ex

        Hx_ey = Pfx @ H_from_Ey
        Hy_ey = Pfy @ H_from_Ey
        Hz_ey = Pfz @ H_from_Ey

        for ir, rx in enumerate(receivers):
            print(f'calculando la impedancia en el receptor {ir:3d}/{len(receivers)} at {rx}, por favor espere :)')
            
            Ex_val = Ex_obs[ir]
            Ey_val = Ey_obs[ir]
            
            Ereceiver[ir, ifreq, 0] = Ex_val
            Ereceiver[ir, ifreq, 1] = Ey_val

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
            
            EH = np.array([[Ex_obs[ir], 0],[0, Ey_obs[ir]]], dtype=complex)

            HH = np.array([[Hx_ex[ir], Hx_ey[ir]],[Hy_ex[ir], Hy_ey[ir]]], dtype=complex)

                
            cond = np.linalg.cond(HH)
            if cond < 1e12:
                Z[ir, ifreq] = np.linalg.solve(HH.T, EH.T).T
            else:
                Z[ir, ifreq] = np.nan
            

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
