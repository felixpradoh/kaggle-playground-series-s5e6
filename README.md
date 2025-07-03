# Kaggle Playground Series S5E6: Predicting Optimal Fertilizers

[![Competition](https://img.shields.io/badge/Kaggle-Competition-blue)](https://www.kaggle.com/competitions/playground-series-s5e6)
[![Status](https://img.shields.io/badge/Status-Active-green)](https://www.kaggle.com/competitions/playground-series-s5e6)

## 📋 Problem Description

This project addresses the **Playground Series - Season 5, Episode 6** on Kaggle, focused on predicting optimal fertilizers for crops based on environmental and soil characteristics.

### Objective
Predict the **most suitable fertilizer type** to maximize crop yield, considering factors such as:
- Weather conditions (temperature, humidity)
- Soil characteristics (type, soil moisture)
- Crop type
- Nutrient levels (nitrogen, potassium, phosphorus)

## 📊 Dataset

### Main Files
- **`train.csv`**: 750,000 training records
- **`test.csv`**: 250,000 records for prediction
- **`sample_submission.csv`**: Required submission format

### Features
| Variable | Description | Type |
|----------|-------------|------|
| `Temparature` | Ambient temperature | Numeric |
| `Humidity` | Relative air humidity | Numeric |
| `Moisture` | Soil moisture | Numeric |
| `Soil Type` | Soil type (Clayey, Sandy, Red, Black) | Categorical |
| `Crop Type` | Crop type (Sugarcane, Millets, Barley, Paddy, etc.) | Categorical |
| `Nitrogen` | Nitrogen level in soil | Numeric |
| `Potassium` | Potassium level in soil | Numeric |
| `Phosphorous` | Phosphorus level in soil | Numeric |

### Target Variable
- **`Fertilizer Name`**: Recommended fertilizer type (e.g., 28-28, 17-17-17, DAP, Urea, etc.)

## 🏗️ Project Structure

```
📦 kaggle-playground-series-s5e6
├── 📁 data/         # Competition datasets and processed data
│   └── processed/   # Preprocessed and feature files
├── 📁 models/       # Trained models and results
│   └── XGB/         # XGBoost model folders
├── 📁 notebooks/    # Jupyter notebooks for EDA and experiments
├── 📁 src/          # Source code and utilities
├── 📁 others/       # Additional notebooks and experiments
```

## 🚀 Collaboration Guide

### Git Workflow

#### Branches

- **`main`**: Main branch with stable code
- **`develop`**: Active development branch
- **`experiment/felix`**: Felix's experiments
- **`experiment/alberto`**: Alberto's experiments

#### Basic Commands

```powershell
# Switch between branches
git checkout develop
git checkout experiment/[name]

# Update your branch with main
git checkout [your-branch]
git merge main
git push origin [your-branch]

# Create a new experiment branch
git checkout -b experiment/[experiment-name]
```

### Naming Conventions

#### Notebook Files

- `kaggle-s5e6-eda.ipynb` - Exploratory Data Analysis
- `kaggle-s5e6-xgboost.ipynb` - XGBoost model experiments
- `kaggle-s5e6-xgboost+optuna.ipynb` - XGBoost with Optuna hyperparameter tuning
- Notebooks in `others/` - Additional experiments and alternative approaches (e.g., `single-xgb.ipynb`, `xgb+original+optuna.ipynb`, etc.)

#### Submission Files

- `submission_[model]_[date]_[score].csv`
- Example: `submission_xgboost_20250601_0845.csv`

### Recommended Workflow

1. **Exploratory Data Analysis (EDA)**
   - Variable distribution
   - Correlations
   - Analysis by soil/crop type
   - Outlier detection

2. **Preprocessing**
   - Encoding categorical variables
   - Normalization/standardization
   - Feature engineering
   - Handling missing values

3. **Modeling**
   - Baseline models
   - Advanced models (XGBoost, LightGBM, etc.)
   - Cross-validation
   - Hyperparameter tuning

4. **Ensemble & Submission**
   - Model combination
   - Final validation
   - Submission generation
