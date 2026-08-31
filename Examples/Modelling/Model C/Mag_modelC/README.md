# Mag_modelC – Forward Magnetics (Torch) | Volcán Puracé

Este ejemplo ejecuta un **forward de magnetometría (intensidad magnética total, IMT)** para el caso de estudio del **Volcán Puracé**, usando un operador implementado en **PyTorch**.

El **modelo `modelC` corresponde al modelo 3D generado en el marco del P8091** a partir de la digitalización de 2 secciones transversales de un modelo geológico de **densidad** obtenido de datos gravimétricos y magnéticos por **Ponce et al. (2024)** e interpolado en **GemPy**; aquí se usa su componente de **susceptibilidad magnética** como modelo de referencia para la simulación directa del campo magnético.

El flujo general es:

- Cargar el **modelo** de susceptibilidad desde `../../../../models/modelC/mag_modelC.npz`
- Cargar **receptores** desde `receivers_modelC.npy`
- Definir el **campo geomagnético** (`geomagnetic_field`, I=24°, D=-5°, B0=29773.7 nT)
- Calcular el **kernel** con `calculateKernelMag`
- Simular la anomalía **IMT** (`K @ chi_active`)
- Graficar el mapa en planta y los cortes Z–X / Y–Z del modelo de susceptibilidad
- Guardar la figura `Modeling_modelC.png` y la respuesta `response_modelC.npy`

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
