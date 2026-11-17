"""Fase 2: evaluación común de los modelos del ganador del punto (sección 2.2).

Todos los modelos pasan por las mismas funciones: se entrenan pliegue por pliegue (un partido fuera, D35),
con los faltantes imputados por la mediana de entrenamiento (D33), y se miden con las mismas métricas.
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.calibration import calibration_curve
from sklearn.impute import SimpleImputer
from sklearn.metrics import (accuracy_score, brier_score_loss, confusion_matrix, f1_score, log_loss,
                             roc_auc_score, roc_curve)
from sklearn.pipeline import make_pipeline

COLOR = '#2a78d6'


def con_imputacion(modelo):
    """Antepone la imputación por mediana: así la mediana sale solo del train de cada pliegue (D33)."""
    return make_pipeline(SimpleImputer(strategy='median'), modelo)


def predecir_cv(modelo, X, y, pliegues):
    """Probabilidad de que gane el equipo 1 para cada punto, predicha por el modelo que no vio su partido."""
    p = pd.Series(np.nan, index=X.index, name='p_equipo_1')
    for train, val in pliegues:
        ajustado = con_imputacion(clone(modelo)).fit(X.iloc[train], y.iloc[train])
        p.iloc[val] = ajustado.predict_proba(X.iloc[val])[:, 1]
    return p


def auc_por_pliegue(y, p, pliegues):
    """AUC dentro de cada partido validado (D36). Los partidos con una sola clase no tienen AUC y se saltan."""
    return pd.Series({i: roc_auc_score(y.iloc[val], p.iloc[val]) for i, (_, val) in enumerate(pliegues)
                      if y.iloc[val].nunique() == 2}, name='AUC')


def metricas(y, p, pliegues=None, umbral=0.5):
    """AUC (media por pliegue si hay pliegues, D36), log-loss y Brier, y accuracy y F1 con el umbral."""
    if pliegues is None:
        auc = {'AUC': roc_auc_score(y, p)}
    else:
        aucs = auc_por_pliegue(y, p, pliegues)
        auc = {'AUC': aucs.mean(), 'AUC_sd': aucs.std(), 'pliegues_con_AUC': len(aucs)}
    return {**auc, 'log_loss': log_loss(y, p), 'Brier': brier_score_loss(y, p),
            'accuracy': accuracy_score(y, p >= umbral), 'F1': f1_score(y, p >= umbral, zero_division=0)}


def _estilo(ax):
    ax.grid(alpha=0.3)
    ax.spines[['top', 'right']].set_visible(False)


def graficar_evaluacion(y, p, nombre, pliegues=None, umbral=0.5):
    """Tres gráficos: AUC por pliegue (en validación cruzada) o curva ROC (en test), calibración y matriz de confusión."""
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.5))
    if pliegues is None:
        fpr, tpr, _ = roc_curve(y, p)
        ax[0].plot(fpr, tpr, color=COLOR)
        ax[0].plot([0, 1], [0, 1], '--', color='gray')
        ax[0].set(title=f'ROC (AUC = {roc_auc_score(y, p):.3f})', xlabel='FPR (falsos positivos)', ylabel='TPR (recall)')
    else:
        aucs = auc_por_pliegue(y, p, pliegues)
        ax[0].bar(range(len(aucs)), aucs.values, color=COLOR)
        ax[0].axhline(0.5, linestyle='--', color='gray')
        ax[0].set(title=f'AUC por partido validado (media = {aucs.mean():.3f})', xlabel='Pliegue (partido fuera)', ylabel='AUC',
                  ylim=(0, 1), xticks=range(len(aucs)), xticklabels=aucs.index + 1)
    real, predicho = calibration_curve(y, p, n_bins=5, strategy='quantile')
    ax[1].plot(predicho, real, 'o-', color=COLOR)
    ax[1].plot([0, 1], [0, 1], '--', color='gray')
    ax[1].set(title='Calibración', xlabel='Probabilidad predicha', ylabel='Proporción real que gana el equipo 1',
              xlim=(0, 1), ylim=(0, 1))
    cm = confusion_matrix(y, p >= umbral, labels=[0, 1])
    ax[2].imshow(cm, cmap='Blues')
    for (i, j), n in np.ndenumerate(cm):
        ax[2].text(j, i, n, ha='center', va='center', fontsize=14, color='white' if n > (cm.max() + cm.min()) / 2 else 'black')
    ax[2].set(title=f'Matriz de confusión (umbral {umbral})', xticks=[0, 1], yticks=[0, 1],
              xticklabels=['pred. gana eq. 2', 'pred. gana eq. 1'], yticklabels=['gana eq. 2', 'gana eq. 1'])
    for a in ax[:2]:
        _estilo(a)
    fig.suptitle(nombre)
    plt.tight_layout()
    plt.show()
