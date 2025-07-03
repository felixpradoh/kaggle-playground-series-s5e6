import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.calibration import calibration_curve

def plot_confusion_matrix(y_true, y_pred, class_names, title="Matriz de Confusión"):
    """
    Genera matriz de confusión con nombres de clases
    """
    plt.figure(figsize=(12, 10))
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names,
                yticklabels=class_names)
    plt.xlabel('Predicción')
    plt.ylabel('Real')
    plt.title(f'{title}\n(Predicción de Fertilizantes)')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()

def plot_probability_analysis(y_pred_proba, classes):
    """
    Análisis de distribución de probabilidades para clasificación multiclase
    """
    plt.figure(figsize=(15, 5))
    
    # Caso multiclase: mostrar probabilidades de la clase más probable
    max_probs = np.max(y_pred_proba, axis=1)
    
    # Subplot 1: Distribución de probabilidades máximas
    plt.subplot(1, 2, 1)
    plt.hist(max_probs, bins=30, alpha=0.7, edgecolor='black')
    plt.title('Distribución de Probabilidades Máximas\n(Confianza del Modelo)')
    plt.xlabel('Probabilidad Máxima')
    plt.ylabel('Frecuencia')
    plt.axvline(1/len(classes), color='red', linestyle='--', 
                label=f'Umbral aleatorio (1/{len(classes)})')
    plt.legend()

    # Subplot 2: Entropía de predicciones (incertidumbre)
    plt.subplot(1, 2, 2)
    entropy = -np.sum(y_pred_proba * np.log(y_pred_proba + 1e-15), axis=1)
    plt.hist(entropy, bins=30, alpha=0.7, edgecolor='black', color='orange')
    plt.title('Distribución de Entropía\n(Incertidumbre del Modelo)')
    plt.xlabel('Entropía')
    plt.ylabel('Frecuencia')
    plt.axvline(np.log(len(classes)), color='red', linestyle='--', 
                label='Máxima incertidumbre')
    plt.legend()

    plt.tight_layout()
    plt.show()

def plot_predictions_vs_actual(y_true, y_pred, fertilizer_encoder, sample_size=100):
    """
    Compara predicciones vs valores reales para una muestra
    """
    # Tomar una muestra aleatoria
    if len(y_true) > sample_size:
        indices = np.random.choice(len(y_true), sample_size, replace=False)
        y_true_sample = y_true.iloc[indices] if hasattr(y_true, 'iloc') else y_true[indices]
        y_pred_sample = y_pred[indices]
    else:
        y_true_sample = y_true
        y_pred_sample = y_pred
    
    # Convertir códigos a nombres
    true_names = [fertilizer_encoder.inverse_transform([code])[0] for code in y_true_sample]
    pred_names = [fertilizer_encoder.inverse_transform([code])[0] for code in y_pred_sample]
    
    # Crear DataFrame para facilitar el ploteo
    comparison_df = pd.DataFrame({
        'Index': range(len(true_names)),
        'Real': true_names,
        'Predicho': pred_names,
        'Correcto': [t == p for t, p in zip(true_names, pred_names)]
    })
    
    plt.figure(figsize=(15, 8))
    
    # Subplot 1: Gráfico de dispersión codificado por colores
    plt.subplot(2, 1, 1)
    colors = ['green' if correct else 'red' for correct in comparison_df['Correcto']]
    plt.scatter(comparison_df['Index'], [1]*len(comparison_df), 
                c=colors, alpha=0.6, s=50)
    plt.title(f'Predicciones vs Valores Reales (Muestra de {len(comparison_df)} casos)\n'
              f'Verde = Correcto, Rojo = Incorrecto')
    plt.xlabel('Índice de Muestra')
    plt.ylabel('Predicción')
    plt.yticks([])
    
    # Agregar estadísticas
    accuracy = sum(comparison_df['Correcto']) / len(comparison_df)
    plt.text(0.02, 0.95, f'Accuracy: {accuracy:.3f}', 
             transform=plt.gca().transAxes, bbox=dict(boxstyle="round", facecolor='wheat'))
    
    # Subplot 2: Distribución de fertilizantes
    plt.subplot(2, 1, 2)
    fertilizer_counts = pd.Series(true_names).value_counts()
    plt.bar(range(len(fertilizer_counts)), fertilizer_counts.values, alpha=0.7)
    plt.title('Distribución de Fertilizantes en la Muestra')
    plt.xlabel('Tipo de Fertilizante')
    plt.ylabel('Frecuencia')
    plt.xticks(range(len(fertilizer_counts)), fertilizer_counts.index, rotation=45, ha='right')
    
    plt.tight_layout()
    plt.show()
    
    return comparison_df

def plot_top3_accuracy_analysis(y_true, y_pred_top3, fertilizer_encoder, sample_size=50):
    """
    Análisis visual de las predicciones TOP-3
    """
    # Tomar muestra
    if len(y_true) > sample_size:
        indices = np.random.choice(len(y_true), sample_size, replace=False)
        y_true_sample = y_true.iloc[indices] if hasattr(y_true, 'iloc') else y_true[indices]
        y_pred_top3_sample = [y_pred_top3[i] for i in indices]
    else:
        y_true_sample = y_true
        y_pred_top3_sample = y_pred_top3
    
    # Análisis de aciertos por posición
    position_hits = [0, 0, 0]  # [pos1, pos2, pos3]
    total_hits = 0
    
    results = []
    for i, (true_code, top3_pred) in enumerate(zip(y_true_sample, y_pred_top3_sample)):
        true_name = fertilizer_encoder.inverse_transform([true_code])[0]
        top3_names = [fertilizer_encoder.inverse_transform([code])[0] for code in top3_pred]
        
        hit_position = -1
        if true_code in top3_pred:
            hit_position = top3_pred.index(true_code)
            position_hits[hit_position] += 1
            total_hits += 1
        
        results.append({
            'Sample': i,
            'True': true_name,
            'Top3': top3_names,
            'Hit': true_code in top3_pred,
            'Position': hit_position
        })
    
    # Visualización
    plt.figure(figsize=(15, 10))
    
    # Subplot 1: Aciertos por posición
    plt.subplot(2, 2, 1)
    positions = ['1st', '2nd', '3rd']
    plt.bar(positions, position_hits, color=['gold', 'silver', '#CD7F32'])
    plt.title('Aciertos por Posición en TOP-3')
    plt.ylabel('Número de Aciertos')
    for i, v in enumerate(position_hits):
        plt.text(i, v + 0.5, str(v), ha='center', va='bottom')
    
    # Subplot 2: MAP@3 vs Accuracy simple
    plt.subplot(2, 2, 2)
    map3_rate = total_hits / len(y_true_sample)
    simple_accuracy = position_hits[0] / len(y_true_sample)
    
    metrics = ['TOP-1 Accuracy', 'TOP-3 Coverage']
    values = [simple_accuracy, map3_rate]
    colors = ['lightcoral', 'lightblue']
    
    bars = plt.bar(metrics, values, color=colors)
    plt.title('Comparación de Métricas')
    plt.ylabel('Tasa de Acierto')
    plt.ylim(0, 1)
    
    for bar, value in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{value:.3f}', ha='center', va='bottom')
    
    # Subplot 3: Distribución de confianza
    plt.subplot(2, 1, 2)
    sample_indices = list(range(min(20, len(results))))
    hit_colors = ['green' if results[i]['Hit'] else 'red' for i in sample_indices]
    
    plt.scatter(sample_indices, [1]*len(sample_indices), c=hit_colors, s=100, alpha=0.7)
    plt.title('Primeras 20 Predicciones: Verde = TOP-3 Hit, Rojo = Miss')
    plt.xlabel('Índice de Muestra')
    plt.ylabel('Resultado')
    plt.yticks([])
    
    # Mostrar nombres en el eje x
    sample_names = [results[i]['True'][:10] + '...' if len(results[i]['True']) > 10 
                   else results[i]['True'] for i in sample_indices]
    plt.xticks(sample_indices, sample_names, rotation=45, ha='right')
    
    plt.tight_layout()
    plt.show()
    
    print(f"\n📊 Resumen TOP-3 Analysis:")
    print(f"  • Total samples: {len(results)}")
    print(f"  • TOP-3 Coverage: {map3_rate:.3f} ({total_hits}/{len(y_true_sample)})")
    print(f"  • Position 1 hits: {position_hits[0]} ({position_hits[0]/len(y_true_sample):.3f})")
    print(f"  • Position 2 hits: {position_hits[1]} ({position_hits[1]/len(y_true_sample):.3f})")
    print(f"  • Position 3 hits: {position_hits[2]} ({position_hits[2]/len(y_true_sample):.3f})")
    
    return pd.DataFrame(results)


def plot_feature_importance(feature_importances, feature_names, title="Análisis de Importancia de Features"):
    """
    Visualiza la importancia de features del modelo RandomForest
    
    Parameters:
    -----------
    feature_importances : array
        Array con las importancias de features del modelo
    feature_names : list
        Lista con los nombres de las features
    title : str
        Título para los gráficos
    
    Returns:
    --------
    feature_importance_df : DataFrame
        DataFrame con las importancias ordenadas
    """
    
    # Crear DataFrame con importancias
    feature_importance_df = pd.DataFrame({
        'feature': feature_names, 
        'importance': feature_importances
    }).sort_values(by='importance', ascending=False)

    # Mostrar top features en consola
    print("🏆 TOP 10 FEATURES MÁS IMPORTANTES:")
    print("=" * 50)
    for i, (_, row) in enumerate(feature_importance_df.head(10).iterrows(), 1):
        print(f"{i:2d}. {row['feature']:<30} {row['importance']:.4f}")
    print("=" * 50)

    # Crear visualización
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

    # Gráfico de barras - Top 20
    top_20_features = feature_importance_df.head(20)
    sns.barplot(data=top_20_features, x='importance', y='feature', ax=ax1, palette='viridis')
    ax1.set_title('🔝 Top 20 Features - Importancia', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Importancia', fontsize=12)
    ax1.set_ylabel('Features', fontsize=12)

    # Histograma de distribución de importancias
    ax2.hist(feature_importance_df['importance'], bins=30, alpha=0.7, color='skyblue', edgecolor='black')
    ax2.set_title('📊 Distribución de Importancias', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Importancia', fontsize=12)
    ax2.set_ylabel('Frecuencia', fontsize=12)
    ax2.axvline(feature_importance_df['importance'].mean(), color='red', linestyle='--', 
               label=f'Media: {feature_importance_df["importance"].mean():.4f}')
    ax2.legend()

    plt.tight_layout()
    plt.show()

    # Estadísticas de importancia
    print(f"\n📈 Estadísticas de importancia:")
    print(f"  • Feature más importante: {feature_importance_df.iloc[0]['feature']} ({feature_importance_df.iloc[0]['importance']:.4f})")
    print(f"  • Importancia promedio: {feature_importance_df['importance'].mean():.4f}")
    print(f"  • Importancia mediana: {feature_importance_df['importance'].median():.4f}")
    print(f"  • Features con importancia > promedio: {(feature_importance_df['importance'] > feature_importance_df['importance'].mean()).sum()}")
    
    return feature_importance_df