"""
Solucionador del sistema lineal del modelamiento MT.

Este modulo solo cambia QUE biblioteca factoriza la matriz. No toca la fisica,
el ensamble, las condiciones de frontera ni el postproceso: ``_prepare_solver``
sigue armando el mismo sistema reducido

    A_ii x_i = b_i ,    x_b = 0   (Dirichlet homogeneo en el borde)

Backends disponibles:

- ``"superlu"`` : ``scipy.sparse.linalg.factorized`` (SuperLU). Es el
                  comportamiento historico del repositorio: un solo hilo y
                  mucho relleno en mallas 3D.
- ``"pardiso"`` : MKL PARDISO via ``pydiso``. Multihilo, mejor reordenamiento y
                  explota la simetria compleja de A.

Los dos son metodos DIRECTOS: factorizan y sustituyen. No hay tolerancia,
residual objetivo ni criterio de parada, asi que la solucion es la misma salvo
redondeo de punto flotante.

Medido en la malla Toy6 (nE = 86.490, nnz(A) = 1.090.890), por frecuencia:

    SuperLU   367,98 s   ~5 GB
    PARDISO     2,04 s   ~1 GB

y el error relativo maximo del tensor de impedancia resultante frente a la
referencia de SuperLU es 7,4e-10 (el residual propio del sistema, con cualquiera
de los dos backends, es ~4e-8: A esta mal condicionada por las celdas de aire).

Por defecto se usa PARDISO si ``pydiso`` esta instalado; si no, se cae solo a
SuperLU sin romper nada. Para forzar uno::

    MT3D_SOLVER=superlu python mi_script.py

o desde Python::

    from Forward import solvers
    solvers.set_default_backend("superlu")
"""

import os
import warnings

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

__all__ = [
    "make_solver",
    "is_complex_symmetric",
    "set_default_backend",
    "get_default_backend",
]

_DEFAULT_BACKEND = os.environ.get("MT3D_SOLVER", "auto")


def set_default_backend(name):
    """Fija el backend por defecto: "auto", "pardiso" o "superlu"."""
    global _DEFAULT_BACKEND
    _DEFAULT_BACKEND = name


def get_default_backend():
    return _DEFAULT_BACKEND


def _have_pardiso():
    try:
        import pydiso.mkl_solver  # noqa: F401

        return True
    except ImportError:
        return False


def is_complex_symmetric(A, tol=1e-12):
    """
    Comprueba A == A^T (simetria compleja, NO hermitica) con tolerancia relativa.

    La matriz MT ``A = C^T M_{1/mu} C + i*omega*M_sigma`` es simetrica compleja
    por construccion: los productos internos de discretize son matrices de masa
    de Galerkin, simetricas incluso con mu o sigma anisotropos, y ``C^T M C``
    hereda esa simetria. La submatriz principal ``A[free][:, free]`` que arma
    ``_prepare_solver`` tambien la conserva, por ser principal.

    Aun asi conviene verificarlo: PARDISO en modo simetrico lee solo el
    triangulo superior, de modo que una matriz no simetrica se resolveria en
    silencio como si lo fuera. El chequeo cuesta ~15 ms contra ~2 s de
    factorizacion.

    Ojo: A no es hermitica. El termino ``i*omega*M_sigma`` es imaginario puro,
    asi que ``||A - A^H|| / ||A||`` vale ~7e-2. El tipo correcto para PARDISO es
    ``complex_symmetric`` (mtype 6), no ``complex_hermitian_*``.
    """
    A = A.tocsr()
    scale = abs(A).max()
    if scale == 0:
        return True
    return abs(A - A.T).max() / scale <= tol


class _SuperLUSolver:
    """SuperLU de SciPy: el comportamiento historico."""

    name = "superlu"

    def __init__(self, A):
        self._factor = spla.factorized(A.tocsc())

    def __call__(self, b):
        b = np.asarray(b)
        if b.ndim == 1:
            return self._factor(b)
        # ``factorized`` no acepta multiples lados derechos
        return np.column_stack([self._factor(b[:, k]) for k in range(b.shape[1])])


class _PardisoSolver:
    """MKL PARDISO, directo y multihilo."""

    name = "pardiso"

    def __init__(self, A, n_threads=None):
        from pydiso.mkl_solver import (
            MKLPardisoSolver,
            set_mkl_pardiso_threads,
            get_mkl_max_threads,
        )

        if n_threads is None:
            n_threads = os.environ.get("MT3D_SOLVER_THREADS", get_mkl_max_threads())
        set_mkl_pardiso_threads(int(n_threads))

        A = A.tocsr()
        if is_complex_symmetric(A):
            # mtype 6: complejo simetrico. PARDISO solo lee el triangulo
            # superior, hay que entregarselo explicitamente.
            self._solver = MKLPardisoSolver(
                sp.triu(A, format="csr"), matrix_type="complex_symmetric"
            )
        else:
            warnings.warn(
                "A no resulto simetrica compleja; se usa el modo no simetrico "
                "(mas lento, pero correcto).",
                RuntimeWarning,
            )
            self._solver = MKLPardisoSolver(A, matrix_type="complex_nonsymmetric")

    def __call__(self, b):
        return self._solver.solve(np.asarray(b, dtype=np.complex128))


def make_solver(A, backend=None):
    """
    Factoriza ``A`` y devuelve un invocable ``solver(b)``.

    ``b`` puede tener forma ``(n,)`` o ``(n, k)``. La interfaz es la misma que
    la de ``scipy.sparse.linalg.factorized``, para que ``_prepare_solver`` y
    ``_solve_secondary`` no necesiten cambiar.
    """
    name = backend or _DEFAULT_BACKEND
    if name == "auto":
        name = "pardiso" if _have_pardiso() else "superlu"

    if name == "pardiso":
        if not _have_pardiso():
            warnings.warn(
                "Se pidio 'pardiso' pero pydiso no esta instalado "
                "(pip install pydiso); se usa SuperLU.",
                RuntimeWarning,
            )
            return _SuperLUSolver(A)
        return _PardisoSolver(A)

    if name == "superlu":
        return _SuperLUSolver(A)

    raise ValueError(f"Backend desconocido: {name!r}. Use 'auto', 'pardiso' o 'superlu'.")
