# MT_Toy7 – Comparación entre el operador MT propuesto y SimPEG

Este ejemplo presenta una **comparación entre el operador magnetotelúrico desarrollado en este repositorio y el operador de modelamiento magnetotelúrico disponible en SimPEG**.

El experimento **Toy7** utiliza el mismo modelo tridimensional, la misma malla, las mismas frecuencias y la misma ubicación de receptor para ambos operadores, permitiendo comparar directamente las respuestas obtenidas para el tensor de impedancias y el tipper.

## Cuadernos

1. **`MT_ModelToy7.ipynb`** — modelamiento directo con el operador propio (`Forward.MT3D.compute_mt_impedance_tipper`).
   Carga la malla (`../../../../models/Toy7/mesh_Toy7_MT.json`) y el modelo de conductividad (`../../../../models/Toy7/model_Toy7_MT.npy`), simula con `frequency_list = np.logspace(-4, 2, 20)` sobre el receptor `(0.0, 0.0, -53.7)` m, y guarda `impedance_Toy7.npy` y `tipper_Toy7.npy`.
2. **`MT_ModelToy7_SimPEG.ipynb`** — modelamiento directo del mismo caso usando SimPEG, guardando `impedance_SimPEG_Toy7.npy` y `tipper_SimPEG_Toy7.npy`.

El flujo general del cuaderno principal es:

- Construir/cargar el modelo tridimensional de conductividad.
- Calcular el tensor de impedancias y el tipper (`compute_mt_impedance_tipper`).
- Comparar las curvas de resistividad aparente y fase, y la magnitud/fase del tipper, contra la respuesta de SimPEG.
- Generar las figuras de comparación incluidas en esta carpeta.

## Archivos incluidos

- `MT_ModelToy7.ipynb`
- `MT_ModelToy7_SimPEG.ipynb`
- `Modeling_Toy7.png`
- `Resistividad_Fase_Operador-SimPEG.png`
- `Tipper_Operador-SimPEG.png`
- `Tipper_ModelingToy7_SimPEG.png`
- `Comparacion_tipper_operador_vs_SimPEG.png`

## Requisitos

- Python 3.9+
- numpy
- scipy
- discretize
- matplotlib
- SimPEG

Instala las dependencias:

```bash
pip install numpy scipy discretize matplotlib simpeg
```
