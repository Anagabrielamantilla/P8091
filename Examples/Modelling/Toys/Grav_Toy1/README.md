# Grav_Toy1 – Forward Gravity (Toy Model)

Script para calcular y graficar la **anomalía gravimétrica (gz)** en **mGal** a partir de un modelo 3D sintético de **contraste de densidad** y un set de **receptores**.

## Requisitos
- Python 3.x
- `numpy`, `scipy`, `matplotlib`, `torch`
- Módulos del proyecto: `Forward.grav3D.grav3D_8091`, `Forward.utils.calculateKernelGrav`

## Entradas
- `../../../../models/Toy1/Toy1_Grav.npz`: `cell_centers (nC,3)`, `Grav_model (nC,)` en Δρ [g/cm³], `dx,dy,dz`
- `receivers_location.npy`: `obs_xyz (nObs,3)` con `[X,Y,Z]`

## Qué hace
1. Carga el modelo (convirtiendo Δρ de g/cm³ a kg/m³) y los receptores
2. Construye el kernel (`calculateKernelGrav`)
3. Calcula `gz` (`grav3D_8091`, `to_mgal=True`)
4. Grafica el mapa en planta (interpolación dispersa) y los cortes Z–X / Y–Z del modelo de densidad
5. Guarda la figura `Modeling_Toy1.png` y la respuesta `response.npy`

Instala dependencias:
```bash
pip install numpy scipy matplotlib torch
```
