# 📁 Carpeta `models/`

Esta carpeta almacena los modelos entrenados y sus metadatos.

## Estructura recomendada:

```
models/
├── baseline/
│   ├── model.pkl
│   └── metadata.json
├── xgboost/
│   ├── model.pkl
│   ├── feature_importance.csv
│   └── metadata.json
└── ensemble/
    ├── model.pkl
    └── metadata.json
```

## Convenciones de naming:
- `{algoritmo}_{fecha}_{score}.pkl`
- Ejemplo: `xgboost_20250601_0845.pkl`

## Metadatos (metadata.json):
```json
{
    "model_type": "XGBoost",
    "features": ["lista", "de", "features"],
    "hyperparameters": {...},
    "cv_score": 0.845,
    "training_date": "2025-06-01",
    "author": "nombre"
}
```
