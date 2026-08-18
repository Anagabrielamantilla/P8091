# Tutorial – Uso del operador directo MT `MT3D`

Esta carpeta contiene un **tutorial paso a paso** para aprender a usar el operador de modelamiento
directo magnetotelúrico implementado en `Forward/MT3D.py`:

```python
from Forward.MT3D import compute_mt_impedance_tipper
Z, T = compute_mt_impedance_tipper(mesh, sigma, receivers, frequencies, mu=None)
```

A diferencia de los ejemplos de `Examples/Modelling/Toys/`, que aplican el operador a modelos ya
construidos, este cuaderno está pensado para quien **nunca lo ha usado**: explica cada argumento,
cada salida y cada decisión de diseño de la malla, y valida el resultado contra la solución
analítica de un semiespacio.

## Contenido de `Tutorial_MT3D.ipynb`

- Paso 0: teoría mínima (ecuación curl–curl, formulación de campo secundario, dos polarizaciones).
- Paso 1: importar el operador y leer su interfaz (entradas, salidas, convenciones fijas).
- Paso 2: construir la malla 3D (`TensorMesh`), padding y capa de aire.
- Paso 3: definir el modelo de conductividad `sigma` (aire = 1e-8 S/m, superficie en z = 0).
- Paso 4: definir receptores y frecuencias.
- Paso 5: verificar el diseño de la malla con el criterio de skin depth.
- Paso 6: ejecutar el operador sobre un semiespacio homogéneo de 10 Ω·m.
- Paso 7: entender la forma e indexación de `Z` (n_rec, n_freq, 2, 2) y `T` (n_rec, n_freq, 2).
- Paso 8: validación contra la solución analítica 1D (ρa = 10 Ω·m, fase = 45°, tipper = 0).
- Paso 9: caso 3D con un bloque conductor y un perfil de 5 receptores.
- Paso 10: recorrido interno del operador, línea por línea (ensamblaje, LU, fuente, Faraday, Z y T).
- Paso 11: guardar/cargar resultados, errores frecuentes, lista de chequeo y plantilla mínima.

## Figuras generadas

- `tutorial_validacion_semiespacio.png`
- `tutorial_modelo_3D.png`
- `tutorial_respuesta_3D.png`
- `tutorial_tipper_3D.png`

## Resultados guardados

- `tutorial_impedance_3D.npy`, `tutorial_tipper_3D.npy`
- `tutorial_receivers.npy`, `tutorial_frequencies.npy`

## Requisitos

- Python 3.9+
- numpy, scipy, discretize, matplotlib
- torch y geoana (los importa `Forward/utils.py` para las utilidades de gravimetría y magnetometría)
- simpeg NO es necesario para este tutorial

```bash
pip install -r requirements.txt
```

El cuaderno trae en su segunda celda una verificación automática del entorno: lista los paquetes
requeridos con su versión e indica cuáles faltan y cómo instalarlos.

## El cuaderno está guardado ya ejecutado

`Tutorial_MT3D.ipynb` se subió **con las salidas incluidas** (texto, tablas y figuras quedan
embebidos dentro del `.ipynb`). Se puede leer completo, con resultados, sin instalar nada y sin
ejecutar nada: GitHub y nbviewer lo renderizan directamente.

Por eso, al editarlo, **no borrar las salidas** antes de subirlo (evitar "Clear All Outputs",
`nbstripout` o hooks de limpieza en git). Los `.png` y `.npy` de esta carpeta son copias sueltas
de lo mismo, útiles para reutilizar las figuras y los resultados en informes.

## Tiempo de ejecución

La malla del tutorial (9.200 celdas, 5 frecuencias) corre completa en unos 6–8 minutos en un
portátil. El costo lo domina la factorización LU: una por frecuencia. Agregar receptores es
prácticamente gratis.
