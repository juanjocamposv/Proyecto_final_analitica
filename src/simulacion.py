"""Fase 4: simulación de parejas y ranking con incertidumbre (D45–D49).

- 4.1 Del punto al partido: fórmula de llegar primero a 7 y su comprobación punto por punto (D47).
- 4.2 Revisión con los partidos reales.
- 4.3 Ranking de parejas: todos contra todos con réplicas (D48).
- 4.4 Incertidumbre: remuestreo de partidos (D49).
- 4.5 Parejas ideales según su estilo.
"""
import json
import warnings
from contextlib import contextmanager
from itertools import combinations
from math import comb

import joblib
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.base import clone

from . import distribuciones as dis
from .datos import DATA_PROCESSED
from .modelo import Simetrico
from .variables import columnas, columnas_jugador, MODELO_1

PUNTOS = 7                                  # gana la pareja que llega primero a 7 (D47)
FIJAS = ['nivel_padel_1', 'nivel_padel_2', 'estatura_cm']      # del formulario: no se sortean (D40)
COLUMNAS_EQUIPO = columnas_jugador(MODELO_1)                    # orden de las variables del equipo en el modelo
RUTA_ENSAMBLAJE = DATA_PROCESSED / 'ensamblaje.joblib'
LOTE = 200_000                              # filas por predicción, para no llenar la memoria


# ── Configuración del ensamblaje (se guarda en la 2.4, D45) ─────────────────────
def guardar_ensamblaje(ajustes, pesos, ruta=RUTA_ENSAMBLAJE):
    """Guarda, de cada modelo del ensamblaje, el modelo sin entrenar con sus hiperparámetros, sus variables y su peso."""
    config = {n: {'modelo': clone(ajustes[n]['modelo']), 'variables': list(ajustes[n]['variables']), 'peso': float(pesos[n])}
              for n in pesos.index}
    ruta.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(config, ruta)
    return ruta


def leer_ensamblaje(ruta=RUTA_ENSAMBLAJE):
    return joblib.load(ruta)


@contextmanager
def _sin_avisos():
    """Los mismos avisos que la 2.3 ya silencia (SVC(probability=True), D34) y el de scikit-learn cuando sus árboles
    predicen en hilos. Se fijan aquí porque los hilos no siempre heredan el filtro de la 2.3; no cambian resultados."""
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', message='The `probability` parameter was deprecated')
        warnings.filterwarnings('ignore', message='`sklearn.utils.parallel.delayed` should be used')
        yield


class Ensamblaje:
    """Voto ponderado de la 2.4 (D37): p = Σ α × p(modelo), cada modelo simétrico (D30)."""

    def __init__(self, config):
        self.config = config

    def fit(self, tabla):
        with _sin_avisos():
            self.modelos_ = {n: Simetrico(clone(c['modelo'])).fit(tabla[columnas(c['variables'])], tabla['gano'])
                             for n, c in self.config.items()}
        return self

    def predecir(self, X):
        p = np.zeros(len(X))
        with _sin_avisos(), joblib.parallel_config(backend='threading', n_jobs=-1):   # árboles en paralelo; mismo resultado
            for i in range(0, len(X), LOTE):
                lote = X.iloc[i:i + LOTE]
                p[i:i + LOTE] = sum(c['peso'] * self.modelos_[n].predict_proba(lote[columnas(c['variables'])])[:, 1]
                                    for n, c in self.config.items())
        return p


# ── 4.1 Del punto al partido (D47) ─────────────────────────────────────────────
def prob_partido(p, puntos=PUNTOS):
    """P(llegar primero a `puntos`) si cada punto se gana con probabilidad p: gana el último punto y el rival hizo k."""
    p = np.asarray(p, float)
    return sum(comb(puntos - 1 + k, k) * p ** puntos * (1 - p) ** k for k in range(puntos))


def simular_partidos(p, n, rng, puntos=PUNTOS):
    """Juega n partidos punto por punto con probabilidad p por punto; devuelve la proporción que gana la pareja."""
    largo = 2 * puntos - 1                  # en 13 puntos una de las dos parejas ya llegó a 7
    gana = rng.random((n, largo)) < p
    a, b = np.cumsum(gana, axis=1), np.cumsum(~gana, axis=1)
    fin_a = np.where((a == puntos).any(axis=1), (a == puntos).argmax(axis=1), largo)
    fin_b = np.where((b == puntos).any(axis=1), (b == puntos).argmax(axis=1), largo)
    return (fin_a < fin_b).mean()


# ── Jugadores y sorteo (D40, D46) ──────────────────────────────────────────────
def tabla_jugadores(distribuciones_jugador, participaciones):
    """Una fila por jugador con sus variables fijas (formulario) y su distribución de atacante y velocidad."""
    fijas = participaciones.drop_duplicates('codigo').set_index('codigo')[[*FIJAS, 'mujer']]
    return distribuciones_jugador.set_index('codigo').join(fijas).reset_index()


def valores_vel(obs):
    """Los valores observados de vel_media_mps de cada jugador (para quien va por bootstrap)."""
    return {c: g['vel_media_mps'].to_numpy(float) for c, g in obs.groupby('codigo')}


def sortear(jug, idx, rng, valores_vel):
    """Sorteo independiente (D46): atacante ~ Bernoulli(p del jugador) y velocidad de su distribución
    (o de sus propios puntos si va por bootstrap), para cada posición de `idx` (índices de jugador)."""
    atk = rng.random(idx.shape) < jug['p_atacante'].to_numpy()[idx]
    vel = np.empty(idx.shape)
    for j, fila in jug.iterrows():
        donde = idx == j
        k = int(donde.sum())
        if k:
            d = dis.distribucion_vel(fila)
            vel[donde] = d.rvs(k, random_state=rng) if d is not None else rng.choice(valores_vel[fila['codigo']], k)
    return atk.astype(float), vel


def _filas_modelo(jug, a, b, atk, vel):
    """Filas del modelo para parejas a (n, 2) contra b (n, 2) con los valores sorteados de sus 4 jugadores."""
    fijas = jug[FIJAS].to_numpy(float)

    def equipo(par, atk_, vel_):
        f = fijas[par].mean(axis=1)
        return {'nivel_padel_1': f[:, 0], 'nivel_padel_2': f[:, 1], 'atacante': atk_.mean(axis=1),
                'vel_media_mps': vel_.mean(axis=1), 'estatura_cm': f[:, 2]}
    ea, eb = equipo(a, atk[:, :2], vel[:, :2]), equipo(b, atk[:, 2:], vel[:, 2:])
    return pd.DataFrame({f'{lado}_{c}': (ea if lado == 'propio' else eb)[c] for c in COLUMNAS_EQUIPO
                         for lado in ('propio', 'rival')})


def prob_punto(ens, jug, a, b, n, rng, valores_vel):
    """p̄ de cada enfrentamiento a contra b: promedio de p(punto) en n sorteos independientes (D46)."""
    m = len(a)
    idx = np.repeat(np.hstack([a, b]), n, axis=0)                  # (m·n, 4) jugadores
    atk, vel = sortear(jug, idx, rng, valores_vel)
    X = _filas_modelo(jug, idx[:, :2], idx[:, 2:], atk, vel)
    return ens.predecir(X).reshape(m, n).mean(axis=1)


# ── 4.2 Partidos reales ────────────────────────────────────────────────────────
def partidos_reales(participaciones, partidos, jug):
    """Las dos parejas de cada partido (como índices de `jug`) y los puntos que ganó cada una según la traza."""
    p = participaciones[participaciones['partido'].isin(partidos)]
    pos = {c: k for k, c in enumerate(jug['codigo'])}
    eq = p.groupby(['partido', 'equipo'])['codigo'].apply(lambda s: [pos[c] for c in sorted(s.unique())]).unstack('equipo')
    e1 = p[p['equipo'] == 1].drop_duplicates(['partido', 'punto']).groupby('partido')['gano_punto']
    return pd.DataFrame({'pareja_1': eq[1], 'pareja_2': eq[2], 'puntos_1': e1.sum(), 'puntos_2': e1.size() - e1.sum()})


# ── 4.3 Todos contra todos (D48) ───────────────────────────────────────────────
def parejas(jug):
    return np.array(list(combinations(range(len(jug)), 2)))


def enfrentamientos(pares):
    """Todos los pares de parejas sin jugadores en común (cada enfrentamiento una vez)."""
    i, j = np.triu_indices(len(pares), 1)
    libre = ~(pares[i][:, :, None] == pares[j][:, None, :]).any(axis=(1, 2))
    return i[libre], j[libre]


def todos_contra_todos(ens, jug, pares, i, j, n, rng, valores_vel):
    """Un todos contra todos: p̄ por enfrentamiento, P(ganar el partido) con la fórmula (D47) y el puntaje de
    cada pareja = su probabilidad media de ganar contra todas las demás."""
    pbar = prob_punto(ens, jug, pares[i], pares[j], n, rng, valores_vel)
    gana = prob_partido(pbar)
    suma = np.bincount(i, gana, len(pares)) + np.bincount(j, 1 - gana, len(pares))
    cuenta = np.bincount(i, minlength=len(pares)) + np.bincount(j, minlength=len(pares))
    return suma / cuenta, pbar


def intervalo_t(x, nivel=0.95):
    """Media y semiancho del intervalo t de las réplicas (filas = réplicas)."""
    r = x.shape[0]
    return x.mean(axis=0), stats.t.ppf(0.5 + nivel / 2, r - 1) * x.std(axis=0, ddof=1) / np.sqrt(r)


def caracteristicas_parejas(jug, pares):
    """Nombre y estilo de cada pareja: nivel medio (0 = primera vez, 1 = principiante, 2 = intermedio o avanzado),
    estatura media, proporción media de puntos como atacante, diferencia de esa proporción entre los dos y
    velocidad media de sus distribuciones, y cuántas mujeres tiene (no es variable del modelo)."""
    nombre = jug['nombre'].to_numpy()
    nivel = (jug['nivel_padel_1'] + 2 * jug['nivel_padel_2']).to_numpy()
    atk = jug['p_atacante'].to_numpy(float)
    vel = np.array([dis.distribucion_vel(f).mean() if dis.distribucion_vel(f) is not None else np.nan
                    for _, f in jug.iterrows()])
    a, b = pares[:, 0], pares[:, 1]
    return pd.DataFrame({'pareja': [f'{nombre[x]} + {nombre[y]}' for x, y in pares],
                         'nivel_medio': (nivel[a] + nivel[b]) / 2,
                         'estatura_media': (jug['estatura_cm'].to_numpy()[a] + jug['estatura_cm'].to_numpy()[b]) / 2,
                         'atacante_medio': (atk[a] + atk[b]) / 2, 'diferencia_atacante': np.abs(atk[a] - atk[b]),
                         'vel_media': (vel[a] + vel[b]) / 2,
                         'mujeres': jug['mujer'].to_numpy()[a] + jug['mujer'].to_numpy()[b]})


# ── 4.4 Remuestreo de partidos (D49) ───────────────────────────────────────────
def replica_remuestreo(config, tabla, obs, jug, familia, pares, i, j, n, rng):
    """Una réplica: partidos con reemplazo → ensamblaje y distribuciones reajustados → todos contra todos."""
    partidos = np.union1d(tabla['partido'].unique(), obs['partido'].unique())
    muestra = rng.choice(partidos, len(partidos), replace=True)
    t = pd.concat([tabla[tabla['partido'] == p] for p in muestra], ignore_index=True)
    o = pd.concat([obs[obs['partido'] == p] for p in muestra], ignore_index=True)
    ens = Ensamblaje(config).fit(t)
    nueva = dis.ajustar_por_jugador(o, familia).set_index('codigo')
    glob = _fila_global(o, familia)
    j_r = jug.copy()
    for k, fila in j_r.iterrows():                 # quien no quedó en la muestra usa la global (D42)
        fuente = nueva.loc[fila['codigo']] if fila['codigo'] in nueva.index else glob
        for c in ['p_atacante', 'metodo_vel', 'scipy_vel', 'parametros_vel']:
            j_r.at[k, c] = fuente[c]
    puntaje, _ = todos_contra_todos(ens, j_r, pares, i, j, n, rng, valores_vel(o))
    return puntaje


def _fila_global(obs, familia):
    x = obs['vel_media_mps'].to_numpy(float)
    _, par = dis.ajustar(x, familia)
    return pd.Series({'p_atacante': obs['atacante'].mean(), 'metodo_vel': 'global', 'scipy_vel': dis.CANDIDATAS[familia],
                      'parametros_vel': json.dumps([float(v) for v in par])})


def resumen_remuestreo(puntajes):
    """Por pareja: intervalo del 95 % (percentiles) del puntaje y del puesto, y % de réplicas en el puesto 1 y en el top 10."""
    puesto = (-puntajes).argsort(axis=1).argsort(axis=1) + 1
    return pd.DataFrame({
        'puntaje_IC95': [f'[{a:.3f}; {b:.3f}]' for a, b in np.percentile(puntajes, [2.5, 97.5], axis=0).T],
        'puesto_IC95': [f'[{a:.0f}; {b:.0f}]' for a, b in np.percentile(puesto, [2.5, 97.5], axis=0).T],
        'pct_primera': 100 * (puesto == 1).mean(axis=0), 'pct_top10': 100 * (puesto <= 10).mean(axis=0)})
