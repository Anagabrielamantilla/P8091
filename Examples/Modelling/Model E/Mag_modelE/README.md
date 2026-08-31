# Mag_modelE – Forward Magnetics (Torch) | Volcán Cerro Machín

Este ejemplo ejecuta un **forward de magnetometría (intensidad magnética total, IMT)** para el caso de estudio del **Cerro Machín**, usando un operador implementado en **PyTorch**.

El **modelo `modelE` corresponde al modelo 3D de contraste de densidad y susceptibilidad magnética tomado de Moreno et al. (2025)**, obtenido mediante la inversión conjunta de datos de gravimetría y magnetometría adquiridos por **Beltrán (2020)**, y se utiliza aquí su componente de **susceptibilidad magnética** como modelo de referencia para la simulación directa del campo magnético.

El flujo general es:

- Cargar el **modelo** de susceptibilidad desde `../../../../models/modelE/mag_modelE.npz`
- Cargar **receptores** desde `receivers_modelE.npy`
- Definir el **campo geomagnético** (`geomagnetic_field`, I=27°, D=-6°, B0=30568 nT)
- Calcular el **kernel** con `calculateKernelMag`
- Simular la anomalía **IMT** (`K @ chi_active`)
- Graficar el mapa en planta y los cortes Z–X / Y–Z del modelo de susceptibilidad
- Guardar la figura `Modeling_modelE.png` y la respuesta `response_modelE.npy`

## Requisitos
- Python 3.9+
- numpy
- scipy
- matplotlib
- torch

Instala dependencias:
```bash
pip install numpy scipy matplotlib torch
```
