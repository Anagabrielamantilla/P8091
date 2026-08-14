import numpy as np
import sys, os
sys.path.append(os.path.abspath("../"))
from Forward.utils import _prepare_bc_masks
from Forward.utils import _primary_fields
from Forward.utils import _prepare_solver
from Forward.utils import _solve_secondary

mu0 = 4 * np.pi * 1e-7

def compute_mt_E_fields(mesh, sigma, receivers, frequencies, mu=None):
    """
    Calcula el campo eléctrico en las polarizaciones x y z para múltiples frecuencias y receptores
    
    Parámetros
    
    mesh: Grilla 3D del modelo
    sigma: Modelo de conductividad 
    receivers: Ubicación de la(s) antena(s)
    frequencies: Conjunto de frecuencias recibidas por la(s) antena(s)

    Retorna
    -------
    E : array (n_rec, n_freq, 2) campo eléctrico para todos los receptores, las frecuencias y las polarizaciones
    Ex : array (n_rec, n_freq, 0) campo eléctrico para todos los receptores, las frecuencias y la polarización en x
    Ey : array (n_rec, n_freq, 1) campo eléctrico para todos los receptores, las frecuencias y la polarización en y
    """
    if mu is None:
        mu = mu0 * np.ones(mesh.nC)

    receivers = np.atleast_2d(receivers)
    n_rec = receivers.shape[0]
    n_freq = len(frequencies)
    
    Ereceiver = np.zeros((n_rec, n_freq, 2), dtype=complex)
    Efull = np.zeros((n_freq, 2, mesh.nE), dtype=complex)

    C = mesh.edge_curl
    bc_mask, _, _ = _prepare_bc_masks(mesh)

    below_surface = mesh.cell_centers[:, 2] <= 0
    sigma_background = 1e-2

    mu_faces = mesh.aveCC2F @ mu

    nFx, nFy = mesh.nFx, mesh.nFy

    Pex = mesh.get_interpolation_matrix(receivers,"Ex")

    Pey = mesh.get_interpolation_matrix(receivers,"Ey")
    

    for ifreq, freq in enumerate(frequencies):
        print(f'Resolviendo para la frecuencia {ifreq} de {len(frequencies)}')

        omega = 2 * np.pi * freq
        Msigma = mesh.get_edge_inner_product(sigma)
        Mmu_inv = mesh.get_face_inner_product(1 / mu)
        A = C.T @ Mmu_inv @ C + 1j * omega * Msigma

        primary_ex, primary_ey = _primary_fields(mesh, omega, sigma_background)
        solver_data = _prepare_solver(A, bc_mask)

        # RHS
        delta_sigma = np.zeros(mesh.nC)
        delta_sigma[below_surface] = sigma[below_surface] - sigma_background # simpeg trabaja con la diferencia de sigma. Sergio
        Mdelta_sigma = mesh.get_edge_inner_product(delta_sigma)
        
        # Ex polarization
        rhs_ex = -1j * omega * (Mdelta_sigma @ primary_ex) # fuente de simpeg. Sergio
        ex_total = primary_ex + _solve_secondary(rhs_ex, solver_data)

        # Ey polarization
        rhs_ey = -1j * omega * (Mdelta_sigma @ primary_ey) # fuente de simpeg. Sergio
        ey_total = primary_ey + _solve_secondary(rhs_ey, solver_data)

        Efull[ifreq,0,:] = ex_total
        Efull[ifreq,1,:] = ey_total

        for ir, rx in enumerate(receivers):
            print(f'calculando el campo eléctrico con polarización x, y en el receptor {ir:3d}/{len(receivers)} de {rx}, por favor espere :)')
            Ex_val = (Pex @ ex_total)[ir]
            Ey_val = (Pey @ ey_total)[ir]

            Ereceiver[ir, ifreq, 0] = Ex_val
            Ereceiver[ir, ifreq, 1] = Ey_val


    return Ereceiver, Efull
