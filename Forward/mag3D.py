import torch, math
import numpy as np

def mag3D_8091(K: torch.Tensor, chi_active: torch.Tensor):
    
    """
    Modela la respuesta magnética como producto matricial entre el kernel K y el
    vector de susceptibilidades (o parámetro escalar) de celdas activas.

    Parámetros
    ----------
    K : torch.Tensor (nObs, nCv)
        Kernel magnético (filas = observaciones, columnas = celdas activas).
    chi_active : torch.Tensor (nCv,)
        Valores del modelo para celdas activas, en el mismo orden de las columnas de K.

    Retorna
    -------
    d_pred : torch.Tensor (nObs,)
        Datos modelados.
    """
    
    if K.ndim != 2:
        raise ValueError("K debe ser 2D (nObs,nCv).")
    if chi_active.ndim != 1:
        chi_active = chi_active.reshape(-1)
        
    return K @ chi_active