# Mag_modelD – Forward Magnetics (Torch) | Volcán Azufral

Este ejemplo ejecuta un **forward de magnetometría (intensidad magnética total, IMT)** para el caso de estudio del **Volcán Azufral**, usando un operador implementado en **PyTorch**.

El **modelo `modelD` corresponde al modelo 3D construido en el marco del P8091** a partir de la digitalización de 9 secciones transversales de un modelo geológico de **densidad** obtenido por **Ponce (2013)** a partir de la interpretación de anomalías de campos potenciales, e interpolado en **GemPy**; aquí se usa su componente de **susceptibilidad magnética** como modelo de referencia para la simulación directa del campo magnético.

El flujo general es:

- Cargar el **modelo** de susceptibilidad desde `../../../../models/modelD/mag_modelD.npz`
- Cargar **receptores** desde `receivers_modelD.npy`
- Definir el **campo geomagnético** (`geomagnetic_field`, I=24°, D=-2°, B0=30151 nT)
- Calcular el **kernel** con `calculateKernelMag`
- Simular la anomalía **IMT** (`K @ chi_active`)
- Graficar el mapa en planta y los cortes Z–X / Y–Z del modelo de susceptibilidad
- Guardar la figura `Modeling_modelD.png` y la respuesta `response_modelD.npy`

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
