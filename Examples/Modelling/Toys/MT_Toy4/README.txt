# MT_Toy4 – Forward Magnetotelluric (Python)

Este ejemplo ejecuta un **modelamiento directo magnetotelúrico (MT)** sobre un modelo sintético de conductividad empleando el operador electromagnético implementado en este repositorio.

El experimento **Toy4** utiliza un modelo sintético almacenado en archivos externos (`mesh_Toy4_MT.json` y `model_Toy4_MT.npy`), lo que permite evaluar el operador sobre una geometría previamente construida y facilitar su comparación con otros códigos de modelamiento.

El flujo general es:

- Cargar la malla desde `mesh_Toy4_MT.json`.
- Cargar el modelo de conductividad desde `model_Toy4_MT.npy`.
- Definir las frecuencias de simulación.
- Definir los receptores MT.
- Calcular el tensor de impedancias mediante `compute_mt_impedance_tipper`.
- Calcular el tipper.
- Guardar los resultados en:
    - `impedance_Toy4.npy`
    - `tipper_Toy4.npy`
- Graficar las respuestas magnetotelúricas obtenidas.

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