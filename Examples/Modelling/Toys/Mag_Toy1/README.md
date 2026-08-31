# Mag_Toy1 – Forward Magnetics (Toy Model)

Script para calcular y graficar la **anomalía de intensidad magnética total (IMT)** en **nT** a partir de un modelo 3D sintético de **susceptibilidad magnética** y un set de **receptores**.

## Requisitos
- Python 3.x
- `numpy`, `scipy`, `matplotlib`, `torch`
- Módulos del proyecto: `Forward.mag3D.mag3D_8091`, `Forward.utils.geomagnetic_field`, `Forward.utils.calculateKernelMag`

## Entradas
- `../../../../models/Toy1/Toy1_Mag.npz`: `cell_centers (nC,3)`, `Mag_model (nC,)` (susceptibilidad SI), `dx,dy,dz`
- `receivers_location.npy`: `obs_xyz (nObs,3)` con `[X,Y,Z]`

## Qué hace
1. Carga el modelo y los receptores
2. Define el campo geomagnético (`geomagnetic_field`, I=90°, D=0°, B0=50000 nT)
3. Construye el kernel (`calculateKernelMag`)
4. Calcula la anomalía IMT (`K @ chi_active`)
5. Grafica el mapa en planta y los cortes Z–X / Y–Z del modelo de susceptibilidad
6. Guarda la figura `Modeling_Toy1.png` y la respuesta `response_Toy1.npy`

Instala dependencias:
```bash
pip install numpy scipy matplotlib torch
```
