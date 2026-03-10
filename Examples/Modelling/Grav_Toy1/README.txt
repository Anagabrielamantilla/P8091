# Grav_Toy2 – Forward Gravity (Toy Model)

Script para calcular y graficar la **anomalía gravimétrica (gz)** en **mGal** a partir de un modelo 3D de **contraste de densidad** y un set de **receptores**.

## Requisitos
- Python 3.x
- `numpy`, `scipy`, `matplotlib`, `torch`
- Módulos del proyecto: `Forward.grav3D.grav3D_8091`, `Forward.utils.calculateKernelGrav`

## Entradas
- `Toy2_Grav.npz`: `cell_centers (nC,3)`, `Grav_model (nC,)`, `dx,dy,dz`
- `receivers_location.npy`: `obs_xyz (nObs,3)` con `[X,Y,Z]`

## Qué hace
1. Carga el modelo y receptores  
2. Construye el kernel (`calculateKernelGrav`)  
3. Calcula `gz` (`grav3D_8091`, `to_mgal=True`)  
4. Grafica el mapa en planta (interpolación dispersa)