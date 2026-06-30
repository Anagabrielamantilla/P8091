# MT_Toy5 – Forward Magnetotelluric 3D (Python)

Este ejemplo ejecuta un **modelamiento directo magnetotelúrico tridimensional (3D)** utilizando el operador electromagnético desarrollado en este repositorio.

El experimento **Toy5** corresponde a un modelo sintético tridimensional empleado para evaluar el comportamiento del operador frente a variaciones espaciales de conductividad y validar el cálculo simultáneo del tensor de impedancias y del tipper.

El flujo general es:

- Construir la malla tridimensional.
- Definir el modelo de conductividad.
- Definir las frecuencias de simulación.
- Definir la ubicación de los receptores.
- Calcular el tensor de impedancias mediante `compute_mt_impedance_tipper`.
- Calcular el tipper.
- Guardar los resultados en:
    - `impedance_3D.npy`
    - `tipper_3D.npy`
- Visualizar las respuestas magnetotelúricas simuladas.

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