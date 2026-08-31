# Grav_modelD – Forward Gravity (Torch) | Volcán Azufral

Este ejemplo ejecuta un **forward de gravedad (gz)** para el caso de estudio del **Volcán Azufral**, usando un operador implementado en **PyTorch**.

El **modelo `modelD` corresponde al modelo 3D construido en el marco del P8091** a partir de la digitalización de 9 secciones transversales de un modelo geológico de **densidad** obtenido por **Ponce (2013)** a partir de la interpretación de anomalías de campos potenciales, e interpolado en **GemPy**, y se utiliza aquí como modelo de referencia para la simulación directa del campo gravitacional.

El flujo general es:

- Cargar el **modelo** de contraste de densidad Δρ desde `../../../../models/modelD/grav_modelD.npz` (convertido de g/cm³ a kg/m³)
- Cargar **receptores** desde `receivers_modelD.npy`
- Calcular el **kernel geométrico** con `calculateKernelGrav`
- Simular la anomalía **gz** con `grav3D_8091` (`to_mgal=True`)
- Graficar el mapa en planta y los cortes Z–X / Y–Z del modelo de densidad
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
