# MT_Toy7 – Comparación entre el operador MT propuesto y SimPEG

Este ejemplo presenta una **comparación entre el operador magnetotelúrico desarrollado en este repositorio y el operador de modelamiento magnetotelúrico disponible en SimPEG**.

El experimento **Toy7** utiliza el mismo modelo tridimensional, la misma malla, las mismas frecuencias y la misma ubicación de los receptores para ambos operadores, permitiendo comparar directamente las respuestas obtenidas para el tensor de impedancias y el tipper.

El flujo general es:

- Construir el modelo tridimensional de conductividad.
- Ejecutar el modelamiento directo con el operador implementado.
- Ejecutar el modelamiento directo utilizando SimPEG.
- Guardar los resultados:
    - `impedance_Toy7.npy`
    - `tipper_Toy7.npy`
    - `impedance_SimPEG_Toy7.npy`
    - `tipper_SimPEG_Toy7.npy`
- Comparar las curvas de resistividad aparente y fase.
- Comparar la magnitud y fase del tipper.
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
