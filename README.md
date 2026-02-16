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
