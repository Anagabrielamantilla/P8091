import torch, math
import numpy as np

def mag3D_8091(K: torch.Tensor, chi_active: torch.Tensor):
    
    if K.ndim != 2:
        raise ValueError("K debe ser 2D (nObs,nCv).")
    if chi_active.ndim != 1:
        chi_active = chi_active.reshape(-1)
        
    return K @ chi_active