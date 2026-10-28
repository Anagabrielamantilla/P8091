# Interpolación LSGL de gravimetría y magnetometría

Entrena un modelo de grafos (LSGL) con las estaciones de gravimetría y magnetometría y predice la anomalía sobre una malla regular. Todo se hace con un solo script.

## Contenido

| Archivo | Descripción |
|---|---|
| `train_and_grid_lsgl.py` | Script principal: entrenamiento, evaluación en test, predicción en la malla, archivos de salida y figuras |
| `lsgl_core.py` | Modelo, entrenamiento, inferencia y funciones auxiliares |
| `requirements.txt` | Dependencias de Python |

## Instalación

Requiere Python 3.10 o superior.

```bash
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Para usar GPU, instale la versión de PyTorch que corresponda a su CUDA siguiendo https://pytorch.org/get-started/locally/. Sin GPU el script también funciona en CPU, pero tarda más.

## Datos de entrada

Los datos no vienen incluidos. Cree una carpeta `data/` junto al script y coloque en ella los dos archivos de estaciones:

```
data/Estaciones_AGCM_SGC_Gravimetria.txt
data/Estaciones_AGCM_SGC_Magnetometria.txt
```

Ambos son archivos de texto separados por espacios, sin encabezado, con 4 columnas: `X  Y  Z  valor`. X y Y son las coordenadas en metros; el valor es la anomalía en mGal (gravimetría) o en nT (magnetometría).

Si los archivos están en otra carpeta, indíquela con `--data-dir`.

## Uso

```bash
python train_and_grid_lsgl.py                              # gravimetría, 5 % test, malla 100x100
python train_and_grid_lsgl.py --target magnetometry
python train_and_grid_lsgl.py --target gravimetry --test-frac 0.1 --seed 7 --nx 200 --ny 200
python train_and_grid_lsgl.py --data-dir /ruta/datos --out-dir /ruta/salida
```

| Argumento | Por defecto | Descripción |
|---|---|---|
| `--target` | `gravimetry` | Variable a interpolar: `gravimetry` o `magnetometry` |
| `--test-frac` | `0.05` | Fracción de estaciones reservadas para test (RMSE) |
| `--seed` | `123` | Semilla del split train/test |
| `--nx`, `--ny` | `100`, `100` | Resolución de la malla |
| `--epochs` | `5000` | Épocas de entrenamiento |
| `--n-mc` | `300` | Muestras de MC-dropout en la inferencia |
| `--residuals-from` | `all` | `all`: el mapa final respeta todas las estaciones; `train`: no usa las de test |
| `--data-dir` | `./data` | Carpeta con los dos archivos de estaciones |
| `--out-dir` | `./resultados` | Carpeta donde se guardan los resultados |

## Salidas (en `--out-dir`)

- `lsgl_pipeline_grid_<target>_..._final.png/.svg`: mapa final (modelo de grafos más la corrección por residuos en las estaciones).
- `..._modelo.png/.svg`: predicción del modelo sin la corrección.
- `..._std.png/.svg`: incertidumbre (desviación estándar de MC-dropout).
- `lsgl_pipeline_grid_<target>_...csv`: la malla, con las columnas `X, Y, modelo, final, std, dentro_envolvente`.
- `lsgl_pipeline_grid_<target>_...npz`: la misma malla en formato NumPy.
- `lsgl_pipeline_model_<target>_...pt`: el modelo entrenado.

Al terminar, la consola muestra el error en cada estación de test y el RMSE de train y de test.

## Método (resumen)

1. Cada variable se normaliza a [-1, 1]. Las estaciones de gravimetría y magnetometría que están muy cerca entre sí se fusionan en un solo nodo del grafo.
2. El modelo aprende los vecinos del grafo y predice las dos variables a la vez. La variable que no se está interpolando entra como covariable.
3. Para predecir la malla, cada celda se conecta a sus k estaciones vecinas en el grafo aprendido (k=14 por defecto).
4. El mapa final suma a la predicción del modelo los residuos de las estaciones (valor real menos modelo), interpolados linealmente. Así el mapa coincide con cada estación y conserva los picos.
5. Las celdas fuera de la envolvente convexa de las estaciones quedan en blanco.
