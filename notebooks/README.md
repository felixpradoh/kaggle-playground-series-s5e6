# 📁 Carpeta `notebooks/`

Esta carpeta contiene los Jupyter notebooks para análisis y experimentación.

## Convenciones de naming:

### Prefijos numéricos:
- `01_` - EDA (Exploratory Data Analysis)
- `02_` - Preprocessing 
- `03_` - Modeling
- `04_` - Ensemble
- `05_` - Analysis

### Ejemplos:
- `01_eda_initial_exploration.ipynb`
- `02_preprocessing_feature_engineering.ipynb`
- `03_model_xgboost_baseline.ipynb`
- `04_ensemble_voting_classifier.ipynb`
- `05_analysis_feature_importance.ipynb`

## Estructura recomendada:

```
notebooks/
├── 01_eda_template.ipynb          # Template básico
├── 01_eda_felix_exploration.ipynb # EDA personalizado
├── 02_preprocessing_main.ipynb    # Preprocesamiento principal
├── 03_model_baseline.ipynb        # Modelo baseline
└── experiments/                   # Experimentos específicos
    ├── experiment_felix_v1.ipynb
    └── experiment_alberto_v1.ipynb
```

## Buenas prácticas:
- Usar markdown para documentar secciones
- Limpiar outputs antes de hacer commit
- Incluir conclusiones al final de cada notebook
- Usar celdas markdown para explicar el proceso
