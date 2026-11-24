"""Fase 2: partición por partido y evaluación común (2.1) y selección por pasos (2.2)."""

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import accuracy_score, r2_score, roc_auc_score

from .datos import SEMILLA
from .variables import columnas

CADA = 5                      # test sistemático: 1 de cada 5 partidos en orden del torneo (D29)
REMUESTREOS = 1000            # bootstrap por partido para los intervalos del 95 % (D30)


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
def simetrizar(tabla, p):
    """p(A gana) = promedio de p(A contra B) y 1 − p(B contra A): las dos filas de un punto suman 1."""
    s = pd.Series(p, index=tabla.index)
    espejo = s.groupby([tabla['partido'], tabla['punto']]).transform('sum') - s     # la otra fila del punto
    return ((s + 1 - espejo) / 2).to_numpy()


def predecir(modelo, tabla, cols):
    """Validación dejando un partido fuera: cada fila se predice con el modelo entrenado sin su partido."""
    p = np.empty(len(tabla))
    for partido in tabla['partido'].unique():
        fuera = (tabla['partido'] == partido).to_numpy()
        m = clone(modelo).fit(tabla.loc[~fuera, cols], tabla.loc[~fuera, 'gano'])
        p[fuera] = m.predict_proba(tabla.loc[fuera, cols])[:, 1]
    return simetrizar(tabla, p)


def metricas(y, p):
    """R² de las probabilidades (0 = naive, 1 = perfecto), AUC y accuracy con corte en 0,5."""
    return {'r2': r2_score(y, p), 'auc': roc_auc_score(y, p), 'accuracy': accuracy_score(y, p >= 0.5)}


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


def diferencia_r2(tabla, p_nuevo, p_actual, muestras):
    """Diferencia de R² (nuevo − actual) y su intervalo del 95 % con los mismos remuestreos (pareada)."""
    y = tabla['gano'].to_numpy()
    difs = []
    for m in muestras:
        i = _filas(tabla, m)
        difs.append(r2_score(y[i], p_nuevo[i]) - r2_score(y[i], p_actual[i]))
    return r2_score(y, p_nuevo) - r2_score(y, p_actual), np.percentile(difs, [2.5, 97.5])


# ── 2.2 Selección por pasos (D31) ──────────────────────────────────────────────
def seleccion_por_pasos(modelo, tabla, base, grupos, muestras):
    """Parte de `base` y agrega cada grupo si sube el R² de validación; si no, prueba sus variables una por una,
    en orden. Devuelve las variables finales, el registro de pasos y las predicciones de cada paso probado."""
    y = tabla['gano']
    actuales = list(base)
    actual = predecir(modelo, tabla, columnas(actuales))
    predicciones = {'modelo 1': actual}
    pasos = [{'paso': 'modelo 1', 'agrega': ', '.join(base), 'r2': r2_score(y, actual), 'dif_vs_actual': np.nan,
              'dif_IC95': '', 'se_queda': True}]

    def probar(nombre, nuevas):
        nonlocal actual
        p = predecir(modelo, tabla, columnas(actuales + nuevas))
        predicciones[nombre] = p
        dif, (inf, sup) = diferencia_r2(tabla, p, actual, muestras)
        queda = dif > 0
        pasos.append({'paso': nombre, 'agrega': ', '.join(nuevas), 'r2': r2_score(y, p), 'dif_vs_actual': dif,
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
    return actuales, pd.DataFrame(pasos).round(4), predicciones


def coeficientes(modelo, tabla, cols, muestras):
    """Coeficientes de la logística estandarizada con todo el train, odds ratio por 1 desviación estándar
    e intervalo del 95 % de los coeficientes reentrenando en los remuestreos de partidos."""
    def coef(t):
        return clone(modelo).fit(t[cols], t['gano'])[-1].coef_[0]
    boot = np.array([coef(tabla.iloc[_filas(tabla, m)]) for m in muestras])
    c = coef(tabla)
    return pd.DataFrame({'coeficiente': c, 'odds_ratio': np.exp(c), 'IC95_inf': np.percentile(boot, 2.5, axis=0),
                         'IC95_sup': np.percentile(boot, 97.5, axis=0)}, index=cols).round(3)
