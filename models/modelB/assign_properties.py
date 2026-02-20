
import numpy as np
# Load model
model_3d = np.load('Gempy\ModelB.npy')

Res_Value = model_3d.copy().astype(float)

Res_Value[Res_Value==1]=13      #-63
Res_Value[Res_Value==2]=15      #-61
Res_Value[Res_Value==3]=450     #374
Res_Value[Res_Value==4]=250     #174
Res_Value[Res_Value==5]=16      #-60
Res_Value[Res_Value==6]=7       #-69
Res_Value[Res_Value==7]=230     #154
Res_Value[Res_Value==8]=17      #-59
Res_Value[Res_Value==9]=76      #0.0

# orden correcto
nx, ny, nz = Res_Value.shape

dx, dy, dz = [13700/nx, 12750/ny, 8500/nz]
x0, y0, z0 = 4729200, 2047800, -3500

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

Res_model = Res_Value.ravel(order="F")

np.savez_compressed(
    'res_modelB.npz',
    cell_centers=cell_centers,
    Res_model=Res_model,
    dx=dx,
    dy=dy,
    dz=dz
)