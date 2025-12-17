
import numpy as np


# Load model
model_3d = np.load('Gempy\modelD.npy')

Grav_Value = model_3d.copy()/1.


Grav_Value[Grav_Value==1]=-0.25
Grav_Value[Grav_Value==2]=0.0
Grav_Value[Grav_Value==3]=-0.1
Grav_Value[Grav_Value==4]=-0.05
Grav_Value[Grav_Value==5]=-0.15
Grav_Value[Grav_Value==6]=0.05
Grav_Value[Grav_Value==7]=0.15



Grav_model = Grav_Value[:,4,:].copy()
Grav_model = np.flip(Grav_model,axis=0)

nx, ny, nz = 50,50,50
dx, dy, dz = [19433.293/nx, 24959.049/ny, 4530/nz]
x0, y0, z0 = 921104.0, 595952.0, -500

# Centers in each axis
x_cent = x0 + (np.arange(nx) + 0.5) * dx
y_cent = y0 + (np.arange(ny) + 0.5) * dy
z_cent = z0 + (np.arange(nz) + 0.5) * dz

# Mesh centers 3D: shape (nz, ny, nx, 3)
zz, yy, xx = np.meshgrid(z_cent, y_cent, x_cent, indexing="ij")
cell_centers = np.stack((xx, yy, zz), axis=-1)

Grav_model = Grav_Value.copy().T
Grav_model = np.flip(Grav_model, axis=0)  

# Flatten to (343000,)
Grav_model = Grav_model.reshape(-1, order='F')  # (70*70*70,)

# Flatten centers to (343000, 3)
cell_centers = cell_centers.reshape(-1, 3, order='F')


np.savez_compressed(
    'grav_modelD.npz',
    cell_centers=cell_centers,
    Grav_model=Grav_model,
    dx=dx,
    dy=dy,
    dz=dz
)



Mag_Value = model_3d.copy()/1.


Mag_Value[Mag_Value==1]=0.00
Mag_Value[Mag_Value==2]=0.0044
Mag_Value[Mag_Value==3]=-0.00755
Mag_Value[Mag_Value==4]=0.0032
Mag_Value[Mag_Value==5]=-0.014

Mag_Value[Mag_Value==6]=0.00074
Mag_Value[Mag_Value==7]=0.074


Mag_model = Mag_Value[:,4,:].copy()
Mag_model = np.flip(Mag_model,axis=0)

nx, ny, nz = 50,50,50  
dx, dy, dz = [19433.293/nx, 24959.049/ny, 4530/nz]
x0, y0, z0 = 921104.0, 595952.0, -500

# Centers in each axis
x_cent = x0 + (np.arange(nx) + 0.5) * dx
y_cent = y0 + (np.arange(ny) + 0.5) * dy
z_cent = z0 + (np.arange(nz) + 0.5) * dz


# Mesh centers 3D: shape (nz, ny, nx, 3)
zz, yy, xx = np.meshgrid(z_cent, y_cent, x_cent, indexing="ij")
cell_centers = np.stack((xx, yy, zz), axis=-1)

Mag_model = Mag_Value.copy().T
Mag_model = np.flip(Mag_model, axis=0)

# Flatten (343000,)
Mag_model = Mag_model.reshape(-1, order='F')  # (70*70*70,)

# Flatten centers to (343000, 3)
cell_centers = cell_centers.reshape(-1, 3, order='F')


np.savez_compressed(
    'mag_modelD.npz',
    cell_centers=cell_centers,
    Mag_model=Mag_model,
    dx=dx,
    dy=dy,
    dz=dz
)