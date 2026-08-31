# MT_Toy5 – Forward Magnetotelluric 3D (Python)

Este ejemplo ejecuta un **modelamiento directo magnetotelúrico tridimensional (3D)** utilizando el operador electromagnético desarrollado en este repositorio.

El experimento **Toy5** corresponde a un modelo sintético tridimensional empleado para evaluar el comportamiento del operador frente a variaciones espaciales de conductividad y validar el cálculo simultáneo del tensor de impedancias y del tipper.

El flujo general es:

- Cargar la malla desde `../../../../models/Toy5/mesh_Toy5_MT.json`.
- Cargar el modelo de conductividad desde `../../../../models/Toy5/model_Toy5_MT.npy`.
- Definir las frecuencias de simulación (`np.logspace(-4, 2, 7)`).
- Definir la ubicación del receptor.
- Calcular el tensor de impedancias y el tipper mediante `compute_mt_impedance_tipper`.
- Guardar los resultados en:
    - `impedance_3D.npy`
    - `tipper_3D.npy`
- Calcular resistividad y fase aparente (`apparent_resistivity`, `phase_deg`) para las cuatro componentes del tensor de impedancias.
- Visualizar las respuestas magnetotelúricas simuladas (impedancia y tipper: magnitud, fase, parte real e imaginaria).

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
