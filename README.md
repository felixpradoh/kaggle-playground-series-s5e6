# Kaggle Playground Series S5E6: Predicting Optimal Fertilizers

[![Competition](https://img.shields.io/badge/Kaggle-Competition-blue)](https://www.kaggle.com/competitions/playground-series-s5e6)
[![Status](https://img.shields.io/badge/Status-Active-green)](https://www.kaggle.com/competitions/playground-series-s5e6)

## 📋 Descripción del Problema

Este proyecto aborda la **Playground Series - Season 5, Episode 6** de Kaggle, enfocado en la predicción de fertilizantes óptimos para cultivos basándose en condiciones ambientales y características del suelo.

### Objetivo
Predecir el **tipo de fertilizante más adecuado** para maximizar el rendimiento de cultivos específicos, considerando factores como:
- Condiciones climáticas (temperatura, humedad)
- Características del suelo (tipo, humedad del suelo)
- Tipo de cultivo
- Niveles de nutrientes (nitrógeno, potasio, fósforo)

## 📊 Dataset

### Archivos Principales
- **`train.csv`**: 750,000 registros de entrenamiento
- **`test.csv`**: 250,000 registros para predicción
- **`sample_submission.csv`**: Formato de envío requerido

### Características (Features)
| Variable | Descripción | Tipo |
|----------|-------------|------|
| `Temparature` | Temperatura ambiente | Numérica |
| `Humidity` | Humedad relativa del aire | Numérica |
| `Moisture` | Humedad del suelo | Numérica |
| `Soil Type` | Tipo de suelo (Clayey, Sandy, Red, Black) | Categórica |
| `Crop Type` | Tipo de cultivo (Sugarcane, Millets, Barley, Paddy, etc.) | Categórica |
| `Nitrogen` | Nivel de nitrógeno en el suelo | Numérica |
| `Potassium` | Nivel de potasio en el suelo | Numérica |
| `Phosphorous` | Nivel de fósforo en el suelo | Numérica |

### Variable Objetivo
- **`Fertilizer Name`**: Tipo de fertilizante recomendado (ej: 28-28, 17-17-17, DAP, Urea, etc.)

## 🏗️ Estructura del Proyecto

```
📦 kaggle-playground-series-s5e6
├── 📁 data/                    # Datasets de la competencia
│   ├── train.csv              # Datos de entrenamiento
│   ├── test.csv               # Datos de prueba
│   └── sample_submission.csv  # Formato de envío
├── 📁 src/                    # Código fuente
├── 📁 notebooks/              # Jupyter notebooks para EDA y experimentos
├── 📁 models/                 # Modelos entrenados
├── 📁 submissions/            # Archivos de envío a Kaggle
└── README.md                  # Este archivo
```

## 🚀 Guía de Colaboración

### Flujo de Trabajo con Git

#### Ramas del Proyecto
- **`main`**: Rama principal con código estable
- **`develop`**: Rama de desarrollo activo
- **`experiment/felix`**: Experimentos de Felix
- **`experiment/alberto`**: Experimentos de Alberto

#### Comandos Básicos
```powershell
# Cambiar entre ramas
git checkout develop
git checkout experiment/[nombre]

# Actualizar rama con cambios de main
git checkout [tu-rama]
git merge main
git push origin [tu-rama]

# Crear nueva rama para experimento
git checkout -b experiment/[nombre-experimento]
```

### Convenciones de Naming

#### Archivos de Notebooks
- `01_eda_[descripcion].ipynb` - Análisis exploratorio
- `02_preprocessing_[descripcion].ipynb` - Preprocesamiento
- `03_model_[algoritmo].ipynb` - Modelos específicos
- `04_ensemble_[descripcion].ipynb` - Modelos ensemble

#### Archivos de Envío
- `submission_[modelo]_[fecha]_[score].csv`
- Ejemplo: `submission_xgboost_20250601_0845.csv`

### Workflow Recomendado

1. **Análisis Exploratorio (EDA)**
   - Distribución de variables
   - Correlaciones
   - Análisis por tipo de suelo/cultivo
   - Identificación de outliers

2. **Preprocesamiento**
   - Encoding de variables categóricas
   - Normalización/estandarización
   - Feature engineering
   - Tratamiento de valores faltantes

3. **Modelado**
   - Baseline models
   - Modelos avanzados (XGBoost, LightGBM, etc.)
   - Validación cruzada
   - Hyperparameter tuning

4. **Ensemble & Submission**
   - Combinación de modelos
   - Validación final
   - Generación de envío

## 🎯 Métricas de Evaluación

La competencia utiliza **accuracy** como métrica principal de evaluación.

## 🔧 Configuración del Entorno

### Dependencias Recomendadas
```python
pandas
numpy
scikit-learn
xgboost
lightgbm
matplotlib
seaborn
jupyter
```

### Instalación
```powershell
pip install -r requirements.txt  # (crear este archivo)
```

## 📈 Progreso del Equipo

### Experimentos Realizados
- [ ] EDA inicial
- [ ] Baseline model
- [ ] Feature engineering
- [ ] XGBoost/LightGBM
- [ ] Ensemble methods
- [ ] Hyperparameter optimization

### Mejores Scores
| Modelo | Score | Fecha | Rama | Notas |
|--------|-------|--------|------|-------|
| TBD | TBD | TBD | TBD | TBD |

## 📝 Notas y Observaciones

### Insights del Dataset
- Dataset balanceado con múltiples tipos de fertilizantes
- Relación clara entre condiciones ambientales y tipo de fertilizante
- Diferentes cultivos requieren diferentes estrategias de fertilización

### TODO List
- [ ] Crear script de preprocesamiento compartido
- [ ] Implementar validación cruzada estratificada
- [ ] Análisis de importancia de features
- [ ] Optimización de hiperparámetros

## 🤝 Contribuidores

- **Felix** - `experiment/felix`
- **Alberto** - `experiment/alberto`

## 📚 Recursos Útiles

- [Competencia en Kaggle](https://www.kaggle.com/competitions/playground-series-s5e6)
- [Documentación de datos](https://www.kaggle.com/competitions/playground-series-s5e6/data)

---

**¿Preguntas o sugerencias?** Abre un issue o comenta en la rama correspondiente.
