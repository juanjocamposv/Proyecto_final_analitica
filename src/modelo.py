"""Fase 2: evaluación común de los modelos del ganador del punto (sección 2.2), ensamblaje (2.7) y modelo final (2.8).

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


def pesos_log_loss(y, P):
    """Peso de cada modelo (columna de P) proporcional a 1 − log-loss, como el 1 − error del notebook 06 (D46)."""
    ll = pd.Series({k: log_loss(y, P[k]) for k in P.columns})
    return (1 - ll) / (1 - ll).sum()


def pesos_optimos(y, P):
    """Pesos no negativos que suman 1 y minimizan el log-loss del promedio ponderado de las columnas de P (D47).

    Se optimiza sobre el simplex con la parametrización softmax (Nelder-Mead, determinista); los pesos menores que
    0,001 se dejan en 0.
    """
    from scipy.optimize import minimize
    Pv, yv = P.to_numpy(), np.asarray(y)
    softmax = lambda z: np.exp(z - z.max()) / np.exp(z - z.max()).sum()
    objetivo = lambda z: log_loss(yv, np.clip(Pv @ softmax(z), 1e-6, 1 - 1e-6))
    z = minimize(objetivo, np.zeros(P.shape[1]), method='Nelder-Mead',
                 options={'maxiter': 4000, 'xatol': 1e-6, 'fatol': 1e-9}).x
    # Los pesos residuales (< 0,001) quedan en 0 exacto: si no, desempatan al azar puntos con la misma predicción
    w = pd.Series(softmax(z), index=P.columns)
    w[w < 1e-3] = 0.0
    return w / w.sum()


def ensamblar_cv(p_modelos, y, pliegues, pesos='iguales'):
    """Ensamblaje por promedio de probabilidades de validación (soft voting, D46-D47).

    Cada p_modelos[k] ya viene del modelo k entrenado sin el partido validado, así que el promedio en ese
    partido es la predicción del ensamblaje entrenado en el mismo pliegue. pesos: 'iguales' (promedio simple),
    'log_loss' (proporcionales a 1 − log-loss) u 'optimos' (minimizan el log-loss). Los pesos de cada pliegue
    salen solo de las predicciones de los otros partidos, nunca del que se valida. Devuelve la probabilidad y
    los pesos usados en cada pliegue.
    """
    P = pd.DataFrame(p_modelos)
    calcular = {'iguales': lambda yt, Pt: pd.Series(1 / P.shape[1], index=P.columns),
                'log_loss': pesos_log_loss, 'optimos': pesos_optimos}[pesos]
    p = pd.Series(np.nan, index=P.index, name='p_equipo_1')
    usados = []
    for train, val in pliegues:
        w = calcular(y.iloc[train], P.iloc[train])
        p.iloc[val] = P.iloc[val] @ w
        usados.append(w)
    return p, pd.DataFrame(usados).reset_index(drop=True)


class Ensamblaje:
    """Modelo final de la 2.8 (D47-D49): promedio ponderado de las probabilidades de varios modelos.

    Cada modelo lleva su imputación por mediana (D33). Los modelos con peso menor que 0,001 no se entrenan.
    """

    def __init__(self, modelos, pesos):
        self.modelos, self.pesos = modelos, pd.Series(pesos, dtype=float)

    def fit(self, X, y):
        self.ajustados_ = {k: con_imputacion(clone(m)).fit(X, y) for k, m in self.modelos.items() if self.pesos[k] > 1e-3}
        return self

    def predict_proba(self, X):
        w = self.pesos[list(self.ajustados_)]
        p = sum(w[k] * m.predict_proba(X)[:, 1] for k, m in self.ajustados_.items()) / w.sum()
        return np.column_stack([1 - p, p])


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
