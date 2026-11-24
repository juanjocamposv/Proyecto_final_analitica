"""Fase 2: partición y evaluación común (2.1), selección por pasos (2.2), hiperparámetros (2.3) y ensamblaje y test (2.4)."""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import LeaveOneGroupOut, cross_val_predict
from skopt import BayesSearchCV

from .datos import SEMILLA
from .variables import columnas

CADA = 5                      # test sistemático: 1 de cada 5 partidos en orden del torneo (D29)
REMUESTREOS = 1000            # bootstrap por partido para los intervalos del 95 % (D30)
ITERACIONES = 40              # búsqueda bayesiana: evaluaciones por modelo, como el notebook 05 (D33)


# ── 2.1 Partición ──────────────────────────────────────────────────────────────
def opciones_test(tabla, puntos):
    """Las 5 muestras sistemáticas posibles (1 de cada 5 partidos en orden del torneo) y cómo se parecen al resto."""
    partidos = sorted(tabla['partido'].unique())
    cancha = puntos.groupby('partido')['cancha'].first()
    fps = puntos.groupby('partido')['fps'].median().round()
    equipo_1 = tabla[tabla['equipo'] == 1]
    variables = [c for c in tabla.columns if c.startswith('propio_')]
    filas = []
    for k in range(CADA):
        test = partidos[k::CADA]
        en_test, fila_test = equipo_1['partido'].isin(test), tabla['partido'].isin(test)
        dif = (tabla.loc[fila_test, variables].mean() - tabla.loc[~fila_test, variables].mean()) / tabla[variables].std()
        filas.append({
            'arranque': k + 1, 'test': test, 'partidos': len(test), 'pct_puntos': round(100 * en_test.mean(), 1),
            'pct_cancha_2': round(100 * (cancha[test] == 2).mean()), 'fps': sorted(set(fps[test].astype(int))),
            'gana_e1_test': round(100 * equipo_1.loc[en_test, 'gano'].mean()),
            'gana_e1_train': round(100 * equipo_1.loc[~en_test, 'gano'].mean()),
            'dif_estandarizada_media': round(dif.abs().mean(), 2),
        })
    return pd.DataFrame(filas).set_index('arranque')


def elegir_test(opciones, pct_cancha_2_total):
    """Regla del test (D29): ~20 % de los partidos y de los puntos y la misma tasa de victorias del equipo 1 que el
    train (±5 puntos); si quedan varias, la de proporción de cancha 2 más parecida al total."""
    n = opciones['partidos'].max()
    ok = opciones[(opciones['partidos'] == n) & opciones['pct_puntos'].between(18, 25)
                  & ((opciones['gana_e1_test'] - opciones['gana_e1_train']).abs() <= 5)]
    return ok.loc[(ok['pct_cancha_2'] - pct_cancha_2_total).abs().idxmin(), 'test']


# ── 2.1 Evaluación común (D30) ─────────────────────────────────────────────────
def _intercambiar(X):
    """Las mismas filas vistas desde el rival: cada columna propio_ pasa a rival_ y al revés."""
    nombres = {c: 'rival_' + c[len('propio_'):] if c.startswith('propio_') else 'propio_' + c[len('rival_'):]
               for c in X.columns}
    return X.rename(columns=nombres)[X.columns]


class Simetrico(ClassifierMixin, BaseEstimator):
    """Cualquier clasificador, pero con p(A gana) = promedio de p(A contra B) y 1 − p(B contra A) (D30)."""

    def __init__(self, modelo):
        self.modelo = modelo

    def fit(self, X, y):
        self.modelo_ = clone(self.modelo).fit(X, y)
        self.classes_ = self.modelo_.classes_
        return self

    def predict_proba(self, X):
        p = self.modelo_.predict_proba(X)[:, 1]
        q = self.modelo_.predict_proba(_intercambiar(X))[:, 1]
        s = (p + 1 - q) / 2
        return np.column_stack([1 - s, s])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


def predecir(modelo, tabla, cols):
    """Validación dejando un partido fuera (`cross_val_predict` por partido): probabilidad simétrica de cada fila."""
    return cross_val_predict(Simetrico(modelo), tabla[cols], tabla['gano'], groups=tabla['partido'],
                             cv=LeaveOneGroupOut(), method='predict_proba', n_jobs=-1)[:, 1]


def metricas(y, p):
    """AUC (principal), accuracy y F1 con corte en 0,5, como en los notebooks del profesor (D30)."""
    return {'auc': roc_auc_score(y, p), 'accuracy': accuracy_score(y, p >= 0.5), 'f1': f1_score(y, p >= 0.5)}


def remuestreos(partidos, b=REMUESTREOS, semilla=SEMILLA):
    """b muestras de partidos con reemplazo (bootstrap por partido)."""
    rng = np.random.default_rng(semilla)
    return [rng.choice(partidos, size=len(partidos), replace=True) for _ in range(b)]


def _filas(tabla, muestra):
    por_partido = {p: np.flatnonzero(tabla['partido'].to_numpy() == p) for p in np.unique(muestra)}
    return np.concatenate([por_partido[p] for p in muestra])


def resumen(tabla, predicciones, muestras):
    """Métricas de cada modelo con todas las predicciones de validación e intervalo del 95 % por bootstrap de partidos."""
    y = tabla['gano'].to_numpy()
    filas = {}
    for nombre, p in predicciones.items():
        boot = pd.DataFrame([metricas(y[i], p[i]) for i in (_filas(tabla, m) for m in muestras)])
        fila = {}
        for m, v in metricas(y, p).items():
            fila[m] = round(v, 3)
            fila[f'{m}_IC95'] = f'[{boot[m].quantile(0.025):.3f}; {boot[m].quantile(0.975):.3f}]'
        filas[nombre] = fila
    return pd.DataFrame(filas).T


def diferencia_auc(tabla, p_nuevo, p_actual, muestras):
    """Diferencia de AUC (nuevo − actual) y su intervalo del 95 % con los mismos remuestreos (pareada)."""
    y = tabla['gano'].to_numpy()
    difs = []
    for m in muestras:
        i = _filas(tabla, m)
        difs.append(roc_auc_score(y[i], p_nuevo[i]) - roc_auc_score(y[i], p_actual[i]))
    return roc_auc_score(y, p_nuevo) - roc_auc_score(y, p_actual), np.percentile(difs, [2.5, 97.5])


# ── 2.2 Selección por pasos (D31) ──────────────────────────────────────────────
def seleccion_por_pasos(modelo, tabla, base, grupos, muestras):
    """Parte de `base` y agrega cada grupo solo si el intervalo del 95 % de la mejora de AUC queda por encima de 0;
    si no, prueba sus variables una por una, en orden, con la misma regla.
    Devuelve las variables finales, el registro de pasos, las predicciones de cada paso y las del modelo final."""
    y = tabla['gano']
    actuales = list(base)
    actual = predecir(modelo, tabla, columnas(actuales))
    predicciones = {'modelo 1': actual}
    pasos = [{'paso': 'modelo 1', 'agrega': ', '.join(base), 'auc': roc_auc_score(y, actual), 'dif_vs_actual': np.nan,
              'dif_IC95': '', 'se_queda': True}]

    def probar(nombre, nuevas):
        nonlocal actual
        p = predecir(modelo, tabla, columnas(actuales + nuevas))
        predicciones[nombre] = p
        dif, (inf, sup) = diferencia_auc(tabla, p, actual, muestras)
        queda = inf > 0
        pasos.append({'paso': nombre, 'agrega': ', '.join(nuevas), 'auc': roc_auc_score(y, p), 'dif_vs_actual': dif,
                      'dif_IC95': f'[{inf:.3f}; {sup:.3f}]', 'se_queda': queda})
        if queda:
            actuales.extend(nuevas)
            actual = p
        return queda

    for grupo, variables in grupos.items():
        nuevas = [v for v in variables if v not in actuales]
        if nuevas and not probar(f'+ {grupo}', nuevas):
            for v in nuevas:
                probar(f'+ {v}', [v])
    return actuales, pd.DataFrame(pasos).round(4), predicciones, actual


def coeficientes(modelo, tabla, cols, muestras):
    """Coeficientes de la logística estandarizada con todo el train, odds ratio por 1 desviación estándar
    e intervalo del 95 % de los coeficientes reentrenando en los remuestreos de partidos."""
    def coef(t):
        return clone(modelo).fit(t[cols], t['gano'])[-1].coef_[0]
    boot = np.array([coef(tabla.iloc[_filas(tabla, m)]) for m in muestras])
    c = coef(tabla)
    return pd.DataFrame({'coeficiente': c, 'odds_ratio': np.exp(c), 'IC95_inf': np.percentile(boot, 2.5, axis=0),
                         'IC95_sup': np.percentile(boot, 97.5, axis=0)}, index=cols).round(3)


# ── 2.3 Hiperparámetros y pasos por modelo (D32, D33) ──────────────────────────
def buscar(modelo, espacio, tabla, cols):
    """`BayesSearchCV` como en el notebook 05 del profesor (40 evaluaciones, AUC), con un partido fuera por pliegue."""
    espacio = {f'modelo__{k}': v for k, v in espacio.items()}
    busqueda = BayesSearchCV(Simetrico(modelo), espacio, n_iter=ITERACIONES, cv=LeaveOneGroupOut(),
                             scoring='roc_auc', n_jobs=-1, random_state=SEMILLA)
    return busqueda.fit(tabla[cols], tabla['gano'], groups=tabla['partido'])


def ajustar(modelo, espacio, tabla, base, grupos, muestras):
    """Busca los hiperparámetros con las variables de `base` (D33) y luego hace la selección por pasos con ellos (D32)."""
    busqueda = buscar(modelo, espacio, tabla, columnas(base))
    mejor = clone(busqueda.best_estimator_.modelo)
    variables, pasos, _, final = seleccion_por_pasos(mejor, tabla, base, grupos, muestras)
    params = {k.removeprefix('modelo__'): v for k, v in busqueda.best_params_.items()}
    return {'modelo': mejor, 'hiperparametros': params, 'variables': variables, 'pasos': pasos, 'prediccion': final}


# ── 2.4 Ensamblaje y test (D37–D39) ─────────────────────────────────────────────
def pesos_accuracy(tabla, predicciones, nombres):
    """Pesos del voto ponderado del notebook 06 del profesor: α = (1 − error) / suma, con la accuracy de validación (D37)."""
    acierto = pd.Series({n: accuracy_score(tabla['gano'], predicciones[n] >= 0.5) for n in nombres})
    return acierto / acierto.sum()


def ensamblar(predicciones, pesos):
    """Voto ponderado con probabilidades: p(ensamblaje) = Σ α × p(modelo) (D37)."""
    return sum(pesos[n] * predicciones[n] for n in pesos.index)


def predecir_test(modelo, train, test, cols):
    """Entrena con todo el train y da la probabilidad simétrica de cada fila del test (D39)."""
    return Simetrico(modelo).fit(train[cols], train['gano']).predict_proba(test[cols])[:, 1]
