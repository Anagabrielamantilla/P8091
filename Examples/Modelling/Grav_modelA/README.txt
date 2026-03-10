# Grav_modelA – Forward Gravity (Torch) | Cerro Machín

Este ejemplo ejecuta un **forward de gravedad (gz)** para el caso de estudio del **Cerro Machín**, usando un operador implementado en **PyTorch**.

El **modelo `modelA` corresponde al modelo de contraste de densidades obtenido a partir de la inversión gravimétrica desarrollada por Gabriel Moreno**, y se utiliza aquí como modelo de referencia para la simulación directa del campo gravitacional.

El flujo general es:

- Cargar **receptores** desde `receiversCM.npy`
- Cargar **modelo** desde `models/modelA/grav_modelA.npz`
- Calcular el **kernel geométrico** con `calculateKernel`
- Simular la anomalía **gz** con `grav3D_8091`
- Graficar un **mapa en planta** interpolando los datos con `LinearNDInterpolator`

## Requisitos
- Python 3.9+
- numpy
- scipy
- matplotlib
- torch

Instala dependencias:
```bash
pip install numpy scipy matplotlib torch