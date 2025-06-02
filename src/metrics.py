def mapk(y_true, y_pred, k=3):
    """
    Métrica MAP@k: y_true = [etiquetas verdaderas], y_pred = [[pred1, pred2, pred3], ...]
    """
    def apk(actual, predicted, k):
        if actual in predicted[:k]:
            return 1.0 / (predicted[:k].index(actual) + 1)
        return 0.0
    
    return sum(apk(a, p, k) for a, p in zip(y_true, y_pred)) / len(y_true)
