
import numpy as np

# Load model
model_3d = np.load('Gempy\ModelB.npy')

Res_Value = model_3d.copy()/1.

Res_Value[Res_Value==1]=2.45
Res_Value[Res_Value==2]=0.85
Res_Value[Res_Value==3]=1.05
Res_Value[Res_Value==4]=1.16
Res_Value[Res_Value==5]=2.42
Res_Value[Res_Value==6]=2.50
Res_Value[Res_Value==7]=2.20
Res_Value[Res_Value==8]=1.30
Res_Value[Res_Value==9]=1.80

nx, ny, nz = 50,50,50
dx, dy, dz = [13700/nx, 12750/ny, 8500/nz]
x0, y0, z0 = 4729200, 2047800, -3500

# Centers in each axis
x_cent = x0 + (np.arange(nx) + 0.5) * dx
y_cent = y0 + (np.arange(ny) + 0.5) * dy
z_cent = z0 + (np.arange(nz) + 0.5) * dz

# Mesh centers 3D: shape (nz, ny, nx, 3)
zz, yy, xx = np.meshgrid(z_cent, y_cent, x_cent, indexing="ij")
cell_centers = np.stack((xx, yy, zz), axis=-1)

Res_model = Res_Value.copy().T
Res_model = np.flip(Res_model, axis=0)

# Flatten (343000,)
Res_model = Res_model.reshape(-1, order='F')  # (70*70*70,)

# Flatten centers to (343000, 3)
cell_centers = cell_centers.reshape(-1, 3, order='F')

np.savez_compressed(
    'res_modelB.npz',
    cell_centers=cell_centers,
    Res_model=Res_model,
    dx=dx,
    dy=dy,
    dz=dz
)