# =============================================================================
# MODEL UTILITIES - Funciones auxiliares para modelado de ML
# =============================================================================

import pandas as pd
import numpy as np
import pickle
import joblib
import os
import json
import time
from datetime import datetime
from typing import List, Dict, Tuple, Any, Optional

# Scikit-learn imports
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
)
from metrics import mapk


class ModelTrainer:
    """Clase para entrenar y evaluar modelos de ML con funcionalidades estándar"""
    
    def __init__(self, model_name: str, model_abbreviation: str):
        self.model_name = model_name
        self.model_abbreviation = model_abbreviation
        self.model = None
        self.training_time = None
        self.metrics = {}
        
    def load_data(self, data_path: str = '../data/processed') -> Dict[str, pd.DataFrame]:
        """Carga todos los datasets preprocesados"""
        print("📂 Cargando datos preprocesados...")
        
        data = {
            'X_train': pd.read_parquet(f'{data_path}/X_train.parquet'),
            'y_train': pd.read_parquet(f'{data_path}/y_train.parquet').squeeze(),
            'X_val': pd.read_parquet(f'{data_path}/X_val.parquet'),
            'y_val': pd.read_parquet(f'{data_path}/y_val.parquet').squeeze(),
            'X_test': pd.read_parquet(f'{data_path}/X_test.parquet')
        }
        
        # Cargar información adicional
        with open(f'{data_path}/feature_info.pkl', 'rb') as f:
            data['feature_info'] = pickle.load(f)
            
        data['label_encoders'] = joblib.load(f'{data_path}/label_encoders.pkl')
        
        print(f"✅ Datos cargados: Train{data['X_train'].shape}, Val{data['X_val'].shape}, Test{data['X_test'].shape}")
        return data
    
    def validate_features(self, features_to_use: List[str], X_train: pd.DataFrame) -> List[str]:
        """Valida que las features existan en los datos"""
        features_available = [f for f in features_to_use if f in X_train.columns]
        features_missing = [f for f in features_to_use if f not in X_train.columns]
        
        if features_missing:
            print(f"⚠️ Features no encontradas: {len(features_missing)}")
            
        print(f"✅ Features válidas: {len(features_available)}")
        return features_available
    
    def train_model(self, model, X_train: pd.DataFrame, y_train: pd.Series, 
                   features: List[str]) -> Any:
        """Entrena el modelo y mide el tiempo"""
        print(f"🚀 Entrenando {self.model_name}...")
        
        start_time = time.time()
        model.fit(X_train[features], y_train)
        self.training_time = time.time() - start_time
        
        self.model = model
        print(f"✅ Entrenamiento completado en {self.training_time:.2f}s")
        return model
    
    def get_top3_predictions(self, probabilities: np.ndarray, classes: np.ndarray) -> List[List]:
        """Convierte probabilidades en predicciones top-3 para MAP@3"""
        top3_predictions = []
        for prob_row in probabilities:
            top3_indices = prob_row.argsort()[-3:][::-1]
            top3_classes = [classes[i] for i in top3_indices]
            top3_predictions.append(top3_classes)
        return top3_predictions
    
    def evaluate_model(self, X_val: pd.DataFrame, y_val: pd.Series, 
                      features: List[str]) -> Dict[str, float]:
        """Evalúa el modelo en el conjunto de validación"""
        print("📊 Evaluando modelo...")
        
        # Predicciones
        y_pred = self.model.predict(X_val[features])
        y_pred_proba = self.model.predict_proba(X_val[features])
        
        # TOP-3 para MAP@3
        classes = self.model.classes_
        y_pred_top3 = self.get_top3_predictions(y_pred_proba, classes)
        
        # Métricas
        map3_score = mapk(y_val.tolist(), y_pred_top3, k=3)
        
        if len(classes) > 2:
            auc_roc = roc_auc_score(y_val, y_pred_proba, multi_class='ovr', average='macro')
        else:
            auc_roc = roc_auc_score(y_val, y_pred_proba[:, 1])
            
        metrics = {
            'map3_score': map3_score,
            'auc_roc': auc_roc,
            'accuracy': accuracy_score(y_val, y_pred),
            'precision': precision_score(y_val, y_pred, average='weighted', zero_division=0),
            'recall': recall_score(y_val, y_pred, average='weighted', zero_division=0),
            'f1_score': f1_score(y_val, y_pred, average='weighted', zero_division=0),
            'num_classes': len(classes)
        }
        self.metrics = metrics
        print(f"🎯 MAP@3: {map3_score:.4f} | AUC-ROC: {auc_roc:.4f} | Accuracy: {metrics['accuracy']:.4f}")
        
        # También devolver las predicciones para visualizaciones
        self.y_pred_top3 = y_pred_top3
        self.classes = classes
        return metrics
    
    def generate_filename_base(self, map3_score: float) -> str:
        """Genera la nomenclatura base para archivos"""
        map3_str = f"{map3_score:.4f}".replace('.', '')
        return f"{self.model_abbreviation}_MAP@3-{map3_str}"
    
    def save_model_artifacts(self, model, metrics: Dict, hyperparams: Dict,
                        features: List[str], model_dir: str = '../models') -> str:
        """Guarda todos los artefactos del modelo con nomenclatura estandarizada"""
        base_filename = self.generate_filename_base(metrics['map3_score'])
        
        # Crear carpeta específica para este modelo
        model_folder = os.path.join(model_dir, base_filename)
        os.makedirs(model_folder, exist_ok=True)
        
        print(f"💾 Guardando artefactos en: {base_filename}/")
        
        # 1. Modelo
        model_path = os.path.join(model_folder, f'{base_filename}_model.pkl')
        joblib.dump(model, model_path)
        
        # 2. Métricas (redondear valores numéricos a 4 decimales)
        metrics_rounded = {}
        for key, value in metrics.items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                metrics_rounded[key] = round(float(value), 4)
            else:
                metrics_rounded[key] = value
        
        metrics_dict = {
            'model_type': self.model_name,
            'model_abbreviation': self.model_abbreviation,
            'tier': 'TIER_1',
            'target_variable': 'Fertilizer Name',
            **metrics_rounded,
            'features_used': len(features),
            'features_list': features,
            'training_time': round(self.training_time, 4) if self.training_time else None,
            'timestamp': datetime.now().isoformat(),
            'kaggle_competition': 'playground-series-s5e6'        }
        
        # Pickle
        metrics_path = os.path.join(model_folder, f'{base_filename}_metrics.pkl')
        with open(metrics_path, 'wb') as f:
            pickle.dump(metrics_dict, f)
            
        # JSON
        metrics_json_path = os.path.join(model_folder, f'{base_filename}_metrics.json')
        with open(metrics_json_path, 'w', encoding='utf-8') as f:
            json.dump(metrics_dict, f, indent=2, ensure_ascii=False)
        
        # 3. Hiperparámetros
        hparams_dict = {
            'model_type': self.model_name,
            'model_abbreviation': self.model_abbreviation,
            'hyperparameters': hyperparams,
            'features_selected': features,
            'num_features': len(features),
            'map3_score_achieved': round(metrics['map3_score'], 4),
            'timestamp': datetime.now().isoformat()
        }
        
        hparams_path = os.path.join(model_folder, f'{base_filename}_hparams.json')
        with open(hparams_path, 'w', encoding='utf-8') as f:
            json.dump(hparams_dict, f, indent=2, ensure_ascii=False)
        print(f"✅ Artefactos guardados en carpeta: {base_filename}/")
        return base_filename
        
    
    def generate_submission(self, X_test: pd.DataFrame, features: List[str],
                          label_encoder, base_filename: str, 
                          model_dir: str = '../models') -> pd.DataFrame:
        """Genera archivo de submission para Kaggle"""
        print("📤 Generando submission...")
        
        # Cargar sample submission
        sample_submission = pd.read_csv('../data/sample_submission.csv')
        
        # Verificar dimensiones
        if len(sample_submission) != len(X_test):
            raise ValueError(f"Dimensiones no coinciden: sample({len(sample_submission)}) vs test({len(X_test)})")
        
        # Predicciones TOP-3
        y_pred_proba = self.model.predict_proba(X_test[features])
        y_pred_top3 = self.get_top3_predictions(y_pred_proba, self.model.classes_)
        
        # Crear submission
        submission_data = []
        for idx, top3_pred in zip(sample_submission['id'], y_pred_top3):
            top3_names = [label_encoder.inverse_transform([pred])[0] for pred in top3_pred]
            predictions_str = ' '.join(top3_names)
            submission_data.append({
                'id': idx,
                'Fertilizer Name': predictions_str
            })
        
        submission_df = pd.DataFrame(submission_data)
        
        # Usar la carpeta específica del modelo
        model_folder = os.path.join(model_dir, base_filename)
        
        # Guardar CSV
        submission_path = os.path.join(model_folder, f'{base_filename}_submission.csv')
        submission_df.to_csv(submission_path, index=False)
        
        # Guardar info
        submission_info = {
            'model_type': self.model_name,
            'model_abbreviation': self.model_abbreviation,
            'map3_score': round(float(self.metrics['map3_score']), 4),
            'submission_file': f'{base_filename}_submission.csv',
            'num_predictions': len(submission_df),
            'format': 'MAP@3 - Top 3 fertilizer names separated by spaces',
            'target_variable': 'Fertilizer Name',
            'timestamp': datetime.now().isoformat(),
            'kaggle_competition': 'playground-series-s5e6'
        }
        
        submission_info_path = os.path.join(model_folder, f'{base_filename}_submission_info.json')
        with open(submission_info_path, 'w', encoding='utf-8') as f:
            json.dump(submission_info, f, indent=2, ensure_ascii=False)
        
        print(f"✅ Submission guardado en: {base_filename}/")
        return submission_df
    
    def print_summary(self, features: List[str]):
        """Imprime resumen final del modelo"""
        print("\n" + "=" * 60)
        print(f"🎉 RESUMEN - {self.model_name}")
        print("=" * 60)
        print(f"🎯 MAP@3: {self.metrics['map3_score']:.4f}")
        print(f"📊 AUC-ROC: {self.metrics['auc_roc']:.4f}")
        print(f"🎯 Features: {len(features)}")
        print(f"⏱️ Tiempo: {self.training_time:.2f}s")
        print("=" * 60)


def print_feature_selection_summary(features_to_use: List[str], features_available: List[str]):
    """Imprime resumen de selección de features"""
    print(f"🎯 FEATURES SELECCIONADAS: {len(features_available)}")
    for i, feat in enumerate(features_available, 1):
        print(f"  {i:2d}. {feat}")


def print_training_config(params: Dict):
    """Imprime configuración de entrenamiento"""
    print("⚙️ CONFIGURACIÓN:")
    for param, value in params.items():
        print(f"  • {param}: {value}")


def show_prediction_examples(y_val: pd.Series, y_pred_top3: List, 
                           fertilizer_encoder, num_examples: int = 3):
    """Muestra ejemplos de predicciones TOP-3"""
    print(f"\n🔍 EJEMPLOS DE PREDICCIONES TOP-3 ({num_examples} muestras):")
    print("=" * 50)
    
    for i in range(min(num_examples, len(y_val))):
        true_label_code = y_val.iloc[i] if hasattr(y_val, 'iloc') else y_val[i]
        top3_pred_codes = y_pred_top3[i]
        
        true_label_name = fertilizer_encoder.inverse_transform([true_label_code])[0]
        top3_pred_names = [fertilizer_encoder.inverse_transform([code])[0] for code in top3_pred_codes]
        
        correct = "✅" if true_label_code in top3_pred_codes else "❌"
        print(f"Muestra {i+1}: {correct}")
        print(f"  Real: {true_label_name}")
        print(f"  Top-3: {top3_pred_names}")
        print("-" * 30)
