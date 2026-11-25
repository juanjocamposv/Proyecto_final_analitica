"""Fase 3: distribuciones de las variables del modelo que cambian de punto a punto (D40–D44).

- 3.1 Exploración: datos por jugador, diferencias entre jugadores y dependencias para la fase 4.
- 3.2 Ajuste con todos los jugadores: candidatas por máxima verosimilitud, KS y AIC (D43).
- 3.3 Ajuste por jugador y tabla para la fase 4 (D42, D44).
"""
import json
import warnings

import numpy as np
import pandas as pd
from scipy import stats

from .datos import SEMILLA

REMUESTREOS = 1000                              # bootstrap del percentil 90 de cada jugador, como en D30
VARIABLES_F3 = ['atacante', 'vel_media_mps']    # las del ensamblaje que cambian de punto a punto (D40)
MIN_PUNTOS = 10                                 # con menos puntos, el jugador usa la distribución global (D42)
ALFA = 0.05                                     # KS: se rechaza la candidata si p < 0,05 (D43)
# Candidatas (D43): nombre -> distribución de scipy. La Erlang es una gamma con forma entera.
CANDIDATAS = {'normal': 'norm', 'uniforme': 'uniform', 'triangular': 'triang', 'exponencial': 'expon',
              'gamma': 'gamma', 'erlang': 'gamma', 'chi2': 'chi2', 'lognormal': 'lognorm',
              'weibull': 'weibull_min', 'beta': 'beta'}


# ── 3.1 Exploración ────────────────────────────────────────────────────────────
def observaciones(participaciones):
    """Jugador × punto con las variables de video (D41), en el orden de partido y punto."""
    cols = ['partido', 'punto', 'equipo', 'lado', 'codigo', 'nombre', *VARIABLES_F3]
    return participaciones.dropna(subset=VARIABLES_F3)[cols].sort_values(['partido', 'punto']).reset_index(drop=True)


def por_jugador(obs):
    """Puntos, tasa de atacante y velocidad mediana de cada jugador."""
    g = obs.groupby(['codigo', 'nombre'])
    return pd.DataFrame({'puntos': g.size(), 'tasa_atacante': g['atacante'].mean(),
                         'vel_mediana': g['vel_media_mps'].median()}).reset_index()


def diferencias_entre_jugadores(obs):
    """¿Todos los jugadores tienen la misma distribución? Chi² para atacante y Kruskal-Wallis para la velocidad."""
    chi2, p_chi2, _, _ = stats.chi2_contingency(pd.crosstab(obs['codigo'], obs['atacante']))
    h, p_kw = stats.kruskal(*[g['vel_media_mps'] for _, g in obs.groupby('codigo')])
    return pd.DataFrame({'prueba': ['chi² jugador × atacante', 'Kruskal-Wallis entre jugadores'],
                         'estadistico': [chi2, h], 'p_valor': [p_chi2, p_kw]}, index=VARIABLES_F3)


def _companeros(obs, v):
    return obs.pivot_table(index=['partido', 'punto', 'equipo'], columns='lado', values=v).dropna()


def _punto_anterior(obs, v):
    """Correlación de cada jugador entre un punto y el anterior suyo; mediana entre jugadores con variación."""
    r = []
    for _, g in obs.groupby('codigo'):
        x = g[v].to_numpy(float)
        if len(x) >= MIN_PUNTOS and x[:-1].std() > 0 and x[1:].std() > 0:
            r.append(stats.spearmanr(x[:-1], x[1:])[0])
    return np.median(r), len(r)


def dependencias(obs):
    """Spearman de las relaciones que la fase 4 tiene que decidir si conserva al sortear."""
    filas = [{'relacion': 'atacante con vel_media_mps (mismo jugador y punto)',
              'rho': stats.spearmanr(obs['atacante'], obs['vel_media_mps'])[0], 'n': len(obs), 'n_es': 'filas'}]
    for v in VARIABLES_F3:
        c = _companeros(obs, v)
        filas.append({'relacion': f'{v}: compañeros en el mismo punto', 'rho': stats.spearmanr(c['LEFT'], c['RIGHT'])[0],
                      'n': len(c), 'n_es': 'pares'})
    for v in VARIABLES_F3:
        rho, n = _punto_anterior(obs, v)
        filas.append({'relacion': f'{v}: mismo jugador, punto anterior (mediana)', 'rho': rho, 'n': n, 'n_es': 'jugadores'})
    return pd.DataFrame(filas).set_index('relacion')


def por_tercio(obs):
    """Tasa de atacante y velocidad media por tercio del torneo (partidos en orden de número, D29) y si cambian."""
    partidos = np.array_split(np.sort(obs['partido'].unique()), 3)
    tercio = obs['partido'].map({p: i + 1 for i, ps in enumerate(partidos) for p in ps})
    g = obs.groupby(tercio)
    t = pd.DataFrame({'partidos': [f'{ps[0]}–{ps[-1]}' for ps in partidos], 'filas': g.size(),
                      'tasa_atacante': g['atacante'].mean(), 'vel_media': g['vel_media_mps'].mean()})
    t.index.name = 'tercio'
    p_chi2 = stats.chi2_contingency(pd.crosstab(tercio, obs['atacante']))[1]
    p_kw = stats.kruskal(*[x['vel_media_mps'] for _, x in g]).pvalue
    return t, {'atacante (chi²)': p_chi2, 'vel_media_mps (Kruskal-Wallis)': p_kw}


# ── 3.2 Ajuste con todos los jugadores (D43) ───────────────────────────────────
def ajustar(x, familia, **fijos):
    """Máxima verosimilitud con scipy (forma, desplazamiento y escala); `fijos` fija parámetros, p. ej. floc."""
    d = getattr(stats, CANDIDATAS[familia])
    with warnings.catch_warnings():               # scipy avisa cuando prueba parámetros fuera del soporte al optimizar
        warnings.simplefilter('ignore', RuntimeWarning)
        if familia == 'erlang':                   # gamma con forma entera: se redondea la forma y se reajusta
            fijos = {**fijos, 'fa': max(1, round(d.fit(x, **fijos)[0]))}
        par = d.fit(x, **fijos)
    return d(*par), par


def nombres_parametros(familia):
    d = getattr(stats, CANDIDATAS[familia])
    return [*(d.shapes.split(', ') if d.shapes else []), 'desplazamiento', 'escala']


def _texto(familia, par):
    return ', '.join(f'{n}={v:.3g}' for n, v in zip(nombres_parametros(familia), par))


def evaluar(x, familia):
    """Una candidata: parámetros, log-verosimilitud, AIC y p-valor de Kolmogórov-Smirnov."""
    x = np.asarray(x, float)
    dist, par = ajustar(x, familia)
    ll = dist.logpdf(x).sum()
    ok = np.isfinite(ll)
    return {'familia': familia, 'parametros': _texto(familia, par), 'k': len(par), 'log_verosimilitud': ll if ok else -np.inf,
            'AIC': 2 * len(par) - 2 * ll if ok else np.inf, 'p_KS': stats.kstest(x, dist.cdf).pvalue}


def comparar_candidatas(x):
    """Las 10 candidatas ordenadas por AIC (D43)."""
    return pd.DataFrame([evaluar(x, f) for f in CANDIDATAS]).sort_values('AIC').reset_index(drop=True)


def elegir(tabla):
    """D43: la de menor AIC entre las que KS no rechaza; None (bootstrap empírico) si todas se rechazan."""
    ok = tabla[tabla['p_KS'] >= ALFA]
    return ok['familia'].iloc[0] if len(ok) else None


# ── 3.3 Por jugador y tabla para la fase 4 (D42, D44) ──────────────────────────
def ajustar_por_jugador(obs, familia):
    """Una fila por jugador con lo que la fase 4 necesita para sortear sus dos variables.

    - atacante: Bernoulli con p = su proporción de puntos como atacante (D42).
    - vel_media_mps: la familia global con el desplazamiento global y su forma y escala (D44); si KS la
      rechaza en el jugador, bootstrap de sus valores. Con menos de MIN_PUNTOS puntos, las dos van con la global.
    - Revisión: intervalo del 95 % del percentil 90 de sus datos (bootstrap de sus puntos) y si el percentil 90
      de su distribución cae adentro.
    """
    x_global = obs['vel_media_mps'].to_numpy(float)
    p_global = obs['atacante'].mean()
    dist_global, par_global = ajustar(x_global, familia) if familia else (None, None)
    rng = np.random.default_rng(SEMILLA)
    filas = []
    for (codigo, nombre), g in obs.groupby(['codigo', 'nombre']):
        x = g['vel_media_mps'].to_numpy(float)
        fila = {'codigo': codigo, 'nombre': nombre, 'puntos': len(x), 'p90_obs': np.quantile(x, 0.9)}
        if len(x) < MIN_PUNTOS:
            dist, par, p_ks = dist_global, par_global, np.nan
            fila.update(p_atacante=p_global, origen='global', metodo_vel='global' if familia else 'bootstrap')
        else:
            fila.update(p_atacante=g['atacante'].mean(), origen='jugador')
            if familia is None:
                dist, par, p_ks, metodo = None, None, np.nan, 'bootstrap'
            else:
                dist, par = ajustar(x, familia, floc=par_global[-2])
                p_ks = stats.kstest(x, dist.cdf).pvalue
                metodo = 'parametrica' if p_ks >= ALFA else 'bootstrap'
                if metodo == 'bootstrap':
                    dist, par = None, None
            fila['metodo_vel'] = metodo
        inf, sup = np.quantile(np.quantile(x[rng.integers(0, len(x), (REMUESTREOS, len(x)))], 0.9, axis=1), [0.025, 0.975])
        p90_dist = float(dist.ppf(0.9)) if dist is not None else np.nan
        fila.update(familia_vel=familia if dist is not None else '', scipy_vel=CANDIDATAS[familia] if dist is not None else '',
                    parametros_vel=json.dumps([float(v) for v in par]) if dist is not None else '',
                    p_KS_vel=p_ks, p90_IC95=f'[{inf:.3f}; {sup:.3f}]', p90_dist=p90_dist,
                    p90_en_IC=bool(inf <= p90_dist <= sup) if dist is not None else np.nan)
        filas.append(fila)
    cols = ['codigo', 'nombre', 'puntos', 'origen', 'p_atacante', 'metodo_vel', 'familia_vel', 'scipy_vel',
            'parametros_vel', 'p_KS_vel', 'p90_obs', 'p90_IC95', 'p90_dist', 'p90_en_IC']
    return pd.DataFrame(filas)[cols]


def distribucion_vel(fila):
    """Reconstruye la distribución de vel_media_mps de una fila de la tabla por jugador (None si va por bootstrap)."""
    if pd.isna(fila['scipy_vel']) or fila['scipy_vel'] == '':      # al leer el CSV, el vacío llega como NaN
        return None
    return getattr(stats, fila['scipy_vel'])(*json.loads(fila['parametros_vel']))
