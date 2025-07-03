import numpy as np
from typing import List, Union, Any

def mapk_old(y_true, y_pred, k=3):
    """
    Métrica MAP@k: y_true = [etiquetas verdaderas], y_pred = [[pred1, pred2, pred3], ...]
    """
    def apk(actual, predicted, k):
        if actual in predicted[:k]:
            return 1.0 / (predicted[:k].index(actual) + 1)
        return 0.0
    
    return sum(apk(a, p, k) for a, p in zip(y_true, y_pred)) / len(y_true)


def apk(actual: Any, predicted: List[Any], k: int = 3) -> float:
    """
    Calcula Average Precision at k para una muestra individual.
    
    Args:
        actual: Etiqueta verdadera
        predicted: Lista de predicciones ordenadas por confianza (mayor a menor)
        k: Número de predicciones a considerar
        
    Returns:
        float: Score AP@k para esta muestra (0.0 a 1.0)
    """
    if not predicted or k <= 0:
        return 0.0
    
    # Implementación optimizada: para al encontrar el primer match
    for i in range(min(k, len(predicted))):
        if predicted[i] == actual:
            return 1.0 / (i + 1)
    
    return 0.0


def mapk(y_true: List[Any], y_pred: List[List[Any]], k: int = 3) -> float:
    """
    Métrica MAP@k: y_true = [etiquetas verdaderas], y_pred = [[pred1, pred2, pred3], ...]
    """
    if not y_true or not y_pred:
        return 0.0
    
    if len(y_true) != len(y_pred):
        raise ValueError(f"y_true y y_pred deben tener la misma longitud: {len(y_true)} vs {len(y_pred)}")
    
    return np.mean([apk(actual, predicted, k) for actual, predicted in zip(y_true, y_pred)])


def mapk_notebook(actual, predicted, k=3):
    """
    Compute mean average precision at k (MAP@k) - implementación del notebook.
    
    Parameters:
    actual : array-like, actual class labels
    predicted : array-like, predicted class indices (top k for each sample)
    k : int, number of predictions to consider
    
    Returns:
    float : MAP@k score
    """
    def apk(a, p, k):
        score = 0.0
        for i in range(min(k, len(p))):
            if p[i] == a:
                score += 1.0 / (i + 1)
                break  # Solo la primera predicción correcta cuenta
        return score
    
    return np.mean([apk(a, p, k) for a, p in zip(actual, predicted)])