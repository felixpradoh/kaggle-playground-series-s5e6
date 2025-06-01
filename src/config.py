"""
Configuración del proyecto Kaggle Playground Series S5E6
"""

import os
from pathlib import Path

# Rutas del proyecto
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
SUBMISSIONS_DIR = PROJECT_ROOT / "submissions"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"
SRC_DIR = PROJECT_ROOT / "src"

# Archivos de datos
TRAIN_FILE = DATA_DIR / "train.csv"
TEST_FILE = DATA_DIR / "test.csv"
SAMPLE_SUBMISSION_FILE = DATA_DIR / "sample_submission.csv"

# Configuración del modelo
RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5

# Columnas del dataset
NUMERICAL_FEATURES = [
    'Temparature', 'Humidity', 'Moisture', 
    'Nitrogen', 'Potassium', 'Phosphorous'
]

CATEGORICAL_FEATURES = [
    'Soil Type', 'Crop Type'
]

TARGET_COLUMN = 'Fertilizer Name'
ID_COLUMN = 'id'

# Configuración de logging
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
LOG_LEVEL = 'INFO'
