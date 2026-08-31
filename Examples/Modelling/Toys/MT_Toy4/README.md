# MT_Toy4 – Forward Magnetotelluric (Python)

Este ejemplo ejecuta un **modelamiento directo magnetotelúrico (MT)** sobre un modelo sintético de conductividad empleando el operador electromagnético implementado en este repositorio.

El experimento **Toy4** utiliza un modelo sintético almacenado en archivos externos (`mesh_Toy4_MT.json` y `model_Toy4_MT.npy`), lo que permite evaluar el operador sobre una geometría previamente construida y facilitar su comparación con otros códigos de modelamiento.

El flujo general es:

- Cargar la malla desde `../../../../models/Toy4/mesh_Toy4_MT.json`.
- Cargar el modelo de conductividad desde `../../../../models/Toy4/model_Toy4_MT.npy`.
- Definir las frecuencias de simulación (`np.logspace(-4, 2, 7)`).
- Definir la ubicación del receptor.
- Calcular el tensor de impedancias y el tipper mediante `compute_mt_impedance_tipper`.
- Guardar los resultados en:
    - `impedance_Toy4.npy`
    - `tipper_Toy4.npy`
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
