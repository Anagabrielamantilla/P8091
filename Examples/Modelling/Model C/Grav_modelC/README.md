# Grav_modelC – Forward Gravity (Torch) | Volcán Puracé

Este ejemplo ejecuta un **forward de gravedad (gz)** para el caso de estudio del **Volcán Puracé**, usando un operador implementado en **PyTorch**.

El **modelo `modelC` corresponde al modelo 3D generado en el marco del P8091** a partir de la digitalización de 2 secciones transversales de un modelo geológico de **densidad** obtenido de datos gravimétricos y magnéticos por **Ponce et al. (2024)** e interpolado en **GemPy**, y se utiliza aquí como modelo de referencia para la simulación directa del campo gravitacional.

El flujo general es:

- Cargar el **modelo** de contraste de densidad Δρ desde `../../../../models/modelC/grav_modelC.npz` (convertido de g/cm³ a kg/m³)
- Cargar **receptores** desde `receivers_modelC.npy`
- Calcular el **kernel geométrico** con `calculateKernelGrav`
- Simular la anomalía **gz** con `grav3D_8091` (`to_mgal=True`)
- Graficar el mapa en planta y los cortes Z–X / Y–Z del modelo de densidad
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
