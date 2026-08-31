# Grav_modelE – Forward Gravity (Torch) | Volcán Cerro Machín

Este ejemplo ejecuta un **forward de gravedad (gz)** para el caso de estudio del **Cerro Machín**, usando un operador implementado en **PyTorch**.

El **modelo `modelE` corresponde al modelo 3D de contraste de densidad y susceptibilidad magnética tomado de Moreno et al. (2025)**, obtenido mediante la inversión conjunta de datos de gravimetría y magnetometría adquiridos por **Beltrán (2020)**, y se utiliza aquí su componente de **densidad** como modelo de referencia para la simulación directa del campo gravitacional.

El flujo general es:

- Cargar el **modelo** de contraste de densidad Δρ desde `../../../../models/modelE/grav_modelE.npz` (convertido de g/cm³ a kg/m³)
- Cargar **receptores** desde `receivers_modelE.npy`
- Calcular el **kernel geométrico** con `calculateKernelGrav`
- Simular la anomalía **gz** con `grav3D_8091` (`to_mgal=True`)
- Graficar el mapa en planta y los cortes Z–X / Y–Z del modelo de densidad
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
