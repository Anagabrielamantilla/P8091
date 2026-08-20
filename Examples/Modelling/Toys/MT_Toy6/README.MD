# MT_Toy6 — Operador MT3DZ vs. SimPEG

Comparación del **operador magnetotelúrico propio de este repositorio** (`Forward/MT3DZ.py`)
contra el modelamiento MT de **SimPEG** (`NSEM.Simulation3DPrimarySecondary`).

El experimento **Toy6** corre las dos soluciones sobre exactamente la misma malla, el mismo modelo
3D de conductividad, las mismas frecuencias y el mismo receptor, de modo que el tensor de
impedancias `Z` y el tipper `T` son directamente comparables.

## Configuración del experimento

| Parámetro | Valor |
|---|---|
| Malla | `../../../../models/Toy6/mesh_Toy6_MT.json` — `TensorMesh` 30 × 30 × 30 (27 000 celdas) |
| Modelo de conductividad | `../../../../models/Toy6/model_Toy6_MT.npy` — σ en S/m, celdas de aire = 1e-8 |
| Frecuencias | `np.logspace(-4, 2, 3)` → 1e-4, 1e-1, 1e2 Hz |
| Receptor | `(0.0, 0.0, -53.7)` m |
| Fondo para SimPEG | semiespacio de 2e-3 S/m (campo primario) |

## Cuadernos

Se ejecutan en este orden:

1. **`MT_ModelToy6_MT3DZ.ipynb`** — modelamiento directo con el operador propio
   (`Forward/MT3DZ.py` → `compute_mt_responses`).
   Guarda `impedance_MT3DZ_Toy6.npy`, `tipper_MT3DZ_Toy6.npy` y la figura `MT3DZ_Toy6.png`.
2. **`MT_ModelToy6_SimPEG.ipynb`** — modelamiento directo con SimPEG.
   Guarda `impedance_SimPEG_Toy6.npy`, `tipper_SimPEG_Toy6.npy` y la figura `SimPEG_Toy6.png`.
3. **`MT_Model_MT3DZ_vs_SimPEG.ipynb`** — comparación. No vuelve a correr ningún forward: carga los
   cuatro `.npy`, superpone las curvas, calcula la diferencia relativa y las métricas
   MAE / RMSE / MAPE. Guarda la figura `MT3DZ_vs_SimPEG_Toy6.png`.

## Respuestas guardadas

| Archivo | Forma | Contenido |
|---|---|---|
| `impedance_MT3DZ_Toy6.npy` | `(n_rec, n_freq, 2, 2)` | tensor de impedancias del operador MT3DZ |
| `tipper_MT3DZ_Toy6.npy` | `(n_rec, n_freq, 2)` | tipper `[Tzx, Tzy]` del operador MT3DZ |
| `impedance_SimPEG_Toy6.npy` | `(n_rec, n_freq, 2, 2)` | tensor de impedancias de SimPEG |
| `tipper_SimPEG_Toy6.npy` | `(n_rec, n_freq, 2)` | tipper `[Tzx, Tzy]` de SimPEG |

Los ejes de frecuencia de los cuatro arreglos siguen el orden de `frequency_list`.

## Figuras

| Figura | Contenido |
|---|---|
| `MT3DZ_Toy6.png` | resistividad aparente, fase y tipper del operador MT3DZ |
| `SimPEG_Toy6.png` | resistividad aparente, fase y tipper de SimPEG |
| `MT3DZ_vs_SimPEG_Toy6.png` | las dos respuestas superpuestas + diferencia relativa |

En la figura de comparación el operador MT3DZ se dibuja como banda gruesa translúcida y SimPEG
como línea a guiones encima: donde SimPEG cae dentro de la banda, las dos soluciones coinciden.

## Interpretación

- La diferencia crece hacia las **frecuencias bajas**. A 1e-4 Hz el *skin depth* supera con holgura
  el tamaño del dominio de `mesh_Toy6_MT.json`, así que la respuesta queda dominada por los bordes
  de la malla. Es un límite de la **malla**, no una discrepancia de formulación: para comparar a
  esas frecuencias hay que agrandar el dominio y el *padding*.
- El operador devuelve `NaN` en las frecuencias donde la matriz de campos magnéticos queda mal
  condicionada; esos puntos se descartan en las métricas y no se dibujan.
- En el tipper conviene mirar también la magnitud absoluta: cuando `|T|` es muy pequeño, un error
  relativo grande sigue siendo un error absoluto diminuto.

## Requisitos

Python 3.11 con `numpy`, `scipy`, `discretize`, `matplotlib` y `simpeg`:

```bash
pip install numpy scipy discretize matplotlib simpeg
```

## Nota sobre `_trash/`

`_trash/` guarda las versiones anteriores de los cuadernos, las respuestas intermedias
(`*_New_*`, `*_Toy6` del operador viejo, campos `Efull`/`Hfull`) y las figuras que quedaron
obsoletas en la limpieza de la carpeta. Está en `.gitignore` y no se sube al repositorio; se puede
borrar cuando ya no haga falta.
