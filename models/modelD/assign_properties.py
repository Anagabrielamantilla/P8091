
import numpy as np
# Load model
model_3d = np.load('Gempy\ModelD.npy')

Sucep_Mag_Value = model_3d.copy()/1.

Sucep_Mag_Value[Sucep_Mag_Value==1]= 0.008
Sucep_Mag_Value[Sucep_Mag_Value==2]=-0.000052
Sucep_Mag_Value[Sucep_Mag_Value==3]= 0.007
Sucep_Mag_Value[Sucep_Mag_Value==4]=-0.000092
Sucep_Mag_Value[Sucep_Mag_Value==5]= 0.0199
Sucep_Mag_Value[Sucep_Mag_Value==6]= 0.00

# orden correcto
nx, ny, nz = Sucep_Mag_Value.shape

dx, dy, dz = [19616.91/nx, 25028.7/ny, 6000/nz]
x0, y0, z0 = 4466965.86, 1662477.77, -500

# Centers in each axis
x_cent = x0 + (np.arange(nx) + 0.5) * dx
y_cent = y0 + (np.arange(ny) + 0.5) * dy
z_cent = z0 + (np.arange(nz) + 0.5) * dz

# grid de centros: (nx, ny, nz)
xx, yy, zz = np.meshgrid(x_cent, y_cent, z_cent, indexing="ij")

# Flatten consistente: usa el MISMO order en centros y en propiedad
cell_centers = np.column_stack([
    xx.ravel(order="F"),
    yy.ravel(order="F"),
    zz.ravel(order="F"),
])

Mag_model = Sucep_Mag_Value.ravel(order="F")

np.savez_compressed(
    'mag_modelD.npz',
    cell_centers=cell_centers,
    Mag_model=Mag_model,
    dx=dx,
    dy=dy,
    dz=dz
)


#--------------------------------------------------------------------------


Density_Value = model_3d.copy().astype(float)  # (nx, ny, nz)

Density_Value[Density_Value==1]=-0.45
Density_Value[Density_Value==2]=-0.18
Density_Value[Density_Value==3]= 0.15
Density_Value[Density_Value==4]= 0.0
Density_Value[Density_Value==5]=-0.54
Density_Value[Density_Value==6]= 0.00



# orden correcto
nx, ny, nz = Density_Value.shape


dx, dy, dz = [19616.91/nx, 25028.7/ny, 6000/nz]
x0, y0, z0 = 4466965.86, 1662477.77, -500


x_cent = x0 + (np.arange(nx) + 0.5) * dx
y_cent = y0 + (np.arange(ny) + 0.5) * dy
z_cent = z0 + (np.arange(nz) + 0.5) * dz

# Mesh centers 3D: shape (nz, ny, nx, 3)
xx, yy, zz = np.meshgrid(x_cent, y_cent, z_cent, indexing="ij")

# Flatten consistente: usa el MISMO order en centros y en propiedad
cell_centers = np.column_stack([
    xx.ravel(order="F"),
    yy.ravel(order="F"),
    zz.ravel(order="F"),
])

Grav_model = Density_Value.ravel(order="F")

np.savez_compressed(
    'grav_modelD.npz',
    cell_centers=cell_centers,
    Grav_model=Grav_model,
    dx=dx,
    dy=dy,
    dz=dz
)


