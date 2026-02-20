
import numpy as np
# Load model
model_3d = np.load('Gempy\ModelA.npy')

Sucep_Mag_Value = model_3d.copy().astype(float)

Sucep_Mag_Value[Sucep_Mag_Value==1]=0.085
Sucep_Mag_Value[Sucep_Mag_Value==2]=0.0944
Sucep_Mag_Value[Sucep_Mag_Value==3]=-0.0955
Sucep_Mag_Value[Sucep_Mag_Value==4]=0.0732
Sucep_Mag_Value[Sucep_Mag_Value==5]=-0.014
Sucep_Mag_Value[Sucep_Mag_Value==6]=0.00

# orden correcto
nx, ny, nz = Sucep_Mag_Value.shape

dx, dy, dz = [13700/nx, 12750/ny, 9000/nz]
x0, y0, z0 = 4729200, 2047800, -4000

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
    'mag_modelA.npz',
    cell_centers=cell_centers,
    Mag_model=Mag_model,
    dx=dx,
    dy=dy,
    dz=dz
)

#------------------------------


Density_Value = model_3d.copy().astype(float)  # (nx, ny, nz)


Density_Value[Density_Value==1]=-0.1
Density_Value[Density_Value==2]=-0.29
Density_Value[Density_Value==3]=0.05
Density_Value[Density_Value==4]=0.35
Density_Value[Density_Value==5]=-0.2
Density_Value[Density_Value==6]=0.00

# orden correcto
nx, ny, nz = Density_Value.shape

dx, dy, dz = [13700/nx, 12750/ny, 9000/nz]
x0, y0, z0 = 4729200, 2047800, -4000

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
    'grav_modelA.npz',
    cell_centers=cell_centers,
    Grav_model=Grav_model,
    dx=dx,
    dy=dy,
    dz=dz,
)


#------------------------------------------------------------------

# chequeo correcto para Fortran order en shape (nx, ny, nz)
ix, iy, iz = nx//5, ny//2, nz//5
k = ix + nx*iy + nx*ny*iz   # F-order para (nx, ny, nz)

print("Expected center:", x_cent[ix], y_cent[iy], z_cent[iz])
print("Saved center   :", cell_centers[k])
print("Value 3D vs flat:", Density_Value[ix,iy,iz], Grav_model[k])

print(cell_centers.shape)      # (nC, 3)
print(cell_centers[:55])        # primeras 5 filas (5 celdas)