# Proyecto 045-2025 — Nuevas Tecnologías Computacionales para el Procesamiento e Inversión Conjunta de Gravimetría, Magnetometría y Magnetotelúrica mediante Aprendizaje Profundo Guiado por Principios Físicos para la Caracterización Multicriterio

![fund](https://img.shields.io/badge/Fundby-Minciencias--ANH-red)

Este repositorio contiene el desarrollo de software y documentación asociados al **Proyecto 045-2025** titulado:

> **NUEVAS TECNOLOGÍAS COMPUTACIONALES PARA EL PROCESAMIENTO E INVERSIÓN CONJUNTA DE GRAVIMETRÍA, MAGNETOMETRÍA Y MAGNETOTELÚRICA MEDIANTE APRENDIZAJE PROFUNDO GUIADO POR PRINCIPIOS FÍSICOS PARA LA CARACTERIZACIÓN MULTICRITERIO**

El proyecto es financiado por **MINCIENCIAS** y la **Agencia Nacional de Hidrocarburos (ANH)**, y se desarrolla en alianza con instituciones académicas y de investigación nacionales.

---

## 📌 Descripción del proyecto

Este repositorio agrupa herramientas de cómputo que permiten:

- Procesar y pre-procesar datos de **gravimetría**, **magnetometría** y **magnetotelúrica**.
- Integrar los tres tipos de datos geofísicos mediante modelos que emplean **aprendizaje profundo guiado por principios físicos** (Physics-Guided Deep Learning).
- Desarrollar esquemas de **inversión conjunta** para caracterización multicriterio de la geología del subsuelo.
- Facilitar el manejo de datos masivos usando estrategias de computación de alto rendimiento.

El código está organizado para permitir su ejecución en distintos **modos de uso** (desarrollo, análisis, ejecución de algoritmos de entrenamiento, etc.).

---

## 🚀 Guía rápida de instalación

Se recomienda utilizar **Anaconda** para la gestión de dependencias y entornos.

1. Clona este repositorio:
   ```bash
   git clone https://github.com/tu_usuario/045-2025.git
   ```
2. Crea y activa un entorno con Python 3.10 o superior (reemplaza `<nombre-entorno>` por el nombre que prefieras):
   ```bash
   conda create -n <Nombre-entorno> python=3.10
   conda activate <Nombre-entorno>
   ```

3. Instala las dependencias (ejecutar dentro de la carpeta del repositorio):
   ```bash
   pip install -r requirements.txt
   ```

4. Instala el paquete del proyecto en modo editable (necesario para que los imports de `Forward` funcionen desde cualquier notebook, sin importar su ubicación):
   ```bash
   pip install -e .
   ```

## 🗺️ Correspondencia de modelos geológicos

| Modelo | Zona volcánica asociada | Descripción breve |
|:------:|--------------------------|-------------------|
| **Model A** | **Volcán Cerro Machín** | Modelo 3D construido en el marco del **P8091** a partir de la digitalización de secciones transversales de un modelo geológico de **densidad** obtenido de la inversión gravimétrica reportada por **Beltrán (2020)**. Las secciones fueron interpoladas en **GemPy**, representando capas volcánicas someras, un conducto volcánico, una cámara magmática profunda y un basamento de mayor densidad. |
| **Model B** | **Volcán Cerro Machín** | Modelo 3D desarrollado en el marco del **Proyecto 8091** mediante la digitalización de **7 secciones transversales y paralelas** de un modelo geológico de **resistividad** propuesto por **Herrera (2020)** a partir de inversión magnetotelúrica. Las secciones fueron interpoladas en **GemPy**, obteniendo una representación de cuerpos volcánicos someros, conducto volcánico y zonas de recarga. |
| **Model C** | **Volcán Puracé** | Modelo 3D generado en el marco del **P8091** a partir de la digitalización de **2 secciones transversales** derivadas de un modelo geológico de **densidad** obtenido de datos gravimétricos y magnéticos por **Ponce et al. (2024)**. La interpolación en **GemPy** permitió representar capas volcánicas someras, zonas de alteración, un conducto volcánico y una cámara magmática profunda. |
| **Model D** | **Volcán Azufral** | Modelo 3D construido en el marco del **P8091** mediante la digitalización de **9 secciones transversales** de un modelo geológico de **densidad** obtenido por **Ponce (2013)** a partir de la interpretación de anomalías de campos potenciales. Las secciones fueron interpoladas en **GemPy**, representando capas volcánicas someras, estructuras volcánicas, áreas de alteración hidrotermal y capas sello. |
| **Model E** | **Volcán Cerro Machín** | Modelo 3D de **contraste de densidad y susceptibilidad magnética** tomado de **Moreno et al. (2025)**, obtenido mediante la inversión conjunta de datos de **gravimetría y magnetometría** adquiridos por **Beltrán (2020)**. El modelo representa capas volcánicas someras, un conducto volcánico conectado con una cámara magmática profunda y un basamento de mayor densidad. |

---

## 📂 Organización del Repositorio

```text
P8091/
├── data/                          # Datos geofísicos
│   ├── gravimetria/               # Datos de gravedad (observados y sintéticos)
│   ├── magnetometria/             # Datos magnéticos
│   ├── magnetotelurica/           # Datos MT (impedancias, resistividad aparente, fase)
│   └── processed/                 # Datos preprocesados listos para inversión
│
├── forward/                       # Modelado directo (Forward Modeling)
│   ├── gravity_forward.py         # Operador directo gravimétrico
│   ├── magnetic_forward.py        # Operador directo magnético
│   ├── mt_forward.py              # Operador directo magnetotelúrico
│   └── utils_forward.py           # Funciones auxiliares físicas y numéricas
│
├── inversion/                     # Esquemas de inversión
│   ├── joint_inversion.py         # Inversión conjunta multi-física
│   ├── gravity_inversion.py       # Inversión individual gravimétrica
│   ├── magnetic_inversion.py      # Inversión individual magnética
│   ├── mt_inversion.py            # Inversión individual MT
│   └── regularization.py          # Términos de regularización física
│
├── models/                        # Modelos físicos y redes neuronales
│   ├── neural_fields.py           # Neural fields físicos consistentes
│   ├── physics_guided_nn.py       # Redes profundas guiadas por física
│   ├── loss_functions.py          # Funciones de pérdida (data + física)
│   └── architectures/             # Arquitecturas (MLP, CNN, Fourier features, etc.)
│
├── preprocessing/                 # Preprocesamiento y limpieza
│   ├── filtering.py               # Filtros y reducción de ruido
│   ├── normalization.py           # Normalización y escalamiento
│   └── interpolation.py           # Interpolación y gridding
│
├── training/                      # Entrenamiento de modelos
│   ├── train_joint.py             # Entrenamiento para inversión conjunta
│   ├── train_individual.py        # Entrenamiento por método individual
│   └── scheduler.py               # Estrategias de optimización
│
├── evaluation/                    # Evaluación y métricas
│   ├── metrics.py                 # RMSE, MAE, chi², etc.
│   └── visualization.py           # Visualización de resultados
│
├── notebooks/                     # Jupyter Notebooks de análisis
│   ├── 01_preprocessing.ipynb
│   ├── 02_forward_modeling.ipynb
│   ├── 03_training.ipynb
│   └── 04_joint_inversion.ipynb
│
├── tests/                         # Pruebas unitarias y validación
│
├── environment.yml                # Entorno Conda
└── README.md                      # Documentación principal