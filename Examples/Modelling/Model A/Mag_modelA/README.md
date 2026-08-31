# Mag_modelA – Forward Magnetics (Torch) | Volcán Cerro Machín

Este ejemplo ejecuta un **forward de magnetometría (intensidad magnética total, IMT)** para el caso de estudio del **Cerro Machín**, usando un operador implementado en **PyTorch**.

El **modelo `modelA` corresponde al modelo 3D construido en el marco del P8091** a partir de la digitalización de secciones transversales de un modelo geológico de densidad obtenido de la inversión gravimétrica de **Beltrán (2020)** e interpolado en **GemPy**; aquí se usa su componente de **susceptibilidad magnética** como modelo de referencia para la simulación directa del campo magnético.

El flujo general es:

- Cargar el **modelo** de susceptibilidad desde `../../../../models/modelA/mag_modelA.npz`
- Cargar **receptores** desde `receivers_modelA.npy`
- Definir el **campo geomagnético** (`geomagnetic_field`, I=27°, D=-6°, B0=30568 nT)
- Calcular el **kernel** con `calculateKernelMag`
- Simular la anomalía **IMT** (`K @ chi_active`)
- Graficar el mapa en planta y los cortes Z–X / Y–Z del modelo de susceptibilidad
- Guardar la figura `Modeling_modelA.png` y la respuesta `response_modelA.npy`

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
