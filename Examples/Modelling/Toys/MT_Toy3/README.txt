# MT_Toy3 – Forward Magnetotelluric (Python)

Este ejemplo ejecuta un **modelamiento directo magnetotelúrico (MT)** utilizando el operador electromagnético desarrollado en este repositorio.

El experimento **Toy3** corresponde a un modelo sintético bidimensional diseñado para evaluar la respuesta del tensor de impedancias y del tipper ante un contraste lateral de conductividad.

El flujo general es:

- Definir la malla tensorial.
- Construir el modelo sintético de conductividad.
- Definir las frecuencias de simulación.
- Definir la ubicación de los receptores.
- Calcular el tensor de impedancias mediante `compute_mt_impedance_tipper`.
- Calcular el tipper.
- Guardar los resultados en:
    - `impedance_Toy3_MT.npy`
    - `tipper_Toy3_MT.npy`
- Graficar las curvas de resistividad aparente, fase y tipper.

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