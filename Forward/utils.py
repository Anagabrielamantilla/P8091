import scipy.sparse.linalg as spla
import numpy as np

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


def cost_and_gradient_log_Z(m, mesh, receivers, frequency_list, Z_obs):
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
        frequency_list
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

def cost_and_gradient_Z(sigma, mesh, receivers, frequency_list, Z_obs):
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
        frequency_list
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
