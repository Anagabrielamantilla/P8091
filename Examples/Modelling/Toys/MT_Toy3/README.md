# MT_Toy3 – Forward Magnetotelluric (Python)

Este ejemplo ejecuta un **modelamiento directo magnetotelúrico (MT)** utilizando el operador electromagnético desarrollado en este repositorio.

El experimento **Toy3** corresponde a un modelo sintético de conductividad diseñado para evaluar la respuesta del tensor de impedancias y del tipper.

El flujo general es:

- Cargar la malla desde `../../../../models/Toy3/mesh_Toy3_MT.json`.
- Cargar el modelo de conductividad desde `../../../../models/Toy3/model_Toy3_MT.npy`.
- Definir las frecuencias de simulación (`np.logspace(-4, 2, 7)`).
- Definir la ubicación del receptor.
- Calcular el tensor de impedancias y el tipper mediante `compute_mt_impedance_tipper`.
- Guardar los resultados en:
    - `impedance_Toy3_MT.npy`
    - `tipper_Toy3_MT.npy`
- Graficar las curvas de resistividad aparente, fase y tipper (magnitud, fase, parte real e imaginaria).

## Requisitos

- Python 3.9+
- numpy
- scipy
- discretize
- matplotlib

Instala las dependencias:

```bash
pip install numpy scipy discretize matplotlib
```
