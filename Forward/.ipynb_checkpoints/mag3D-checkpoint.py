import torch, math
import numpy as np

def mag3D_8091(kernel, m):
    # extraer device del m
    # asignar al tensor kernel el device de m
    return (kernel * m.unsqueeze(0)).sum(dim=1) * 1e9