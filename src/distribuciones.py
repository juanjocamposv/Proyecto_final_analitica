"""Fase 3: distribuciones de las variables de estilo y movimiento de cada jugador (D50-D57).

Secciones:
1. Observaciones jugador × punto y componentes que se ajustan (D50, D53).
2. Exploración univariada, bivariada y multivariada (3.1-3.3).
3. Ajuste de las candidatas por MLE, pruebas de bondad y elección (3.4; D52, D54, D56).
4. Ajuste por jugador y bootstrap empírico (3.5; D57).
5. Tabla final para la fase 4 (3.6).
"""
import json
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import integrate, optimize, stats

from .datos import DATA_INTERIM, SEMILLA

COLOR, GRIS, TINTA = '#2a78d6', '#9aa5b1', '#52514e'
VARIABLES_F3 = ['prof_media_m', 'prof_sd_m', 'pct_en_su_lado', 'vel_p90', 'area_m2', 'pct_quieto', 'rafagas_por_min']
BINARIA = 'atacante'
ORDEN_TORNEO = ['partido', 'punto', 'equipo', 'lado']      # orden temporal del torneo (D51)
MIN_OBS_JUGADOR = 10        # por jugador solo se ajusta con ≥ 10 observaciones; si no, la global (D50)
ALFA = 0.05                 # prueba de bondad (D52)
VENTANA_MOVIL = 30          # media móvil de la secuencia (D51)
B_BOOT = 2000               # remuestreos del bootstrap (semilla datos.SEMILLA)
CELDA_AREA_M2 = 0.25        # el área se mide en cuadros de 0,5 × 0,5 m (D31)

# Componentes (D53): nombre -> (variable de origen, tipo, límites de la variable, definición)
COMPONENTES = {
    'prof_media_m': ('prof_media_m', 'continua', (0, 10), 'prof_media_m'),
    'prof_sd_m': ('prof_sd_m', 'continua', (0, None), 'prof_sd_m'),
    'vel_p90': ('vel_p90', 'continua', (0, None), 'vel_p90'),
    'pct_quieto': ('pct_quieto', 'continua', (0, 100), 'pct_quieto'),
    'area_m2': ('area_m2', 'continua', (0, None), 'area_m2 como continua (solo comparación)'),
    'cuadros_area': ('area_m2', 'conteo', (0, None), 'cuadros extra = área / 0,25 − 1 (conteo ≥ 0)'),
    'p_100_su_lado': ('pct_en_su_lado', 'bernoulli', (0, 1), '1 si pct_en_su_lado = 100'),
    'deficit_su_lado': ('pct_en_su_lado', 'continua', (0, 100), '100 − pct_en_su_lado, solo si < 100'),
    'p_0_rafagas': ('rafagas_por_min', 'bernoulli', (0, 1), '1 si rafagas_por_min = 0'),
    'rafagas_positivas': ('rafagas_por_min', 'continua', (0, None), 'rafagas_por_min, solo si > 0'),
    'atacante': ('atacante', 'bernoulli', (0, 1), '1 si es atacante en el punto (D21)'),
}
# area_m2 continua se ajusta solo para comparar: a la fase 4 entra como conteo de cuadros (D53)
SOLO_COMPARACION = {'area_m2'}


def _estilo(ax, eje='y'):
    ax.grid(axis=eje, alpha=0.25)
    ax.spines[['top', 'right']].set_visible(False)


# ══ 1. Observaciones y componentes ══════════════════════════════════════════════
def leer_participaciones():
    """participaciones.csv con los códigos como texto de 6 dígitos (D8)."""
    return pd.read_csv(DATA_INTERIM / 'participaciones.csv', encoding='utf-8-sig',
                       dtype={'codigo': str, 'codigo_companero': str})


def observaciones(participaciones):
    """Jugador × punto con video y alguna variable de la fase 3 (D50), en el orden del torneo (D51)."""
    cols = [*VARIABLES_F3, BINARIA]
    obs = participaciones.loc[participaciones[cols].notna().any(axis=1),
                              ['partido', 'punto', 'equipo', 'lado', 'codigo', 'nombre', 'gano_punto', *cols]]
    return obs.sort_values(ORDEN_TORNEO).reset_index(drop=True)


def valores_componente(obs, comp):
    """Valores de un componente (D53), con la llave del jugador. Los ceros técnicos se tratan en el ajuste (D54)."""
    var, tipo, _, _ = COMPONENTES[comp]
    d = obs[['codigo', var]].dropna()
    x = d[var].astype(float)
    if comp == 'cuadros_area':
        x = (x / CELDA_AREA_M2 - 1).round()
    elif comp == 'p_100_su_lado':
        x = (x == 100).astype(float)
    elif comp == 'deficit_su_lado':
        d, x = d[x < 100], 100 - x[x < 100]
    elif comp == 'p_0_rafagas':
        x = (x == 0).astype(float)
    elif comp == 'rafagas_positivas':
        d, x = d[x > 0], x[x > 0]
    return pd.DataFrame({'codigo': d['codigo'].to_numpy(), 'x': x.to_numpy()})


# ══ 2. Exploración (3.1-3.3) ═══════════════════════════════════════════════════
def _p_estacionariedad(s):
    from statsmodels.tsa.stattools import adfuller, kpss
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')          # KPSS avisa cuando el p-valor sale del rango de su tabla (0,01-0,10)
        return adfuller(s, autolag='AIC')[1], kpss(s, regression='c', nlags='auto')[1]


def resumen_univariado(obs):
    """Forma, dispersión, homogeneidad entre jugadores, estacionariedad e independencia de cada variable (3.1)."""
    from statsmodels.stats.diagnostic import acorr_ljungbox
    from statsmodels.tsa.stattools import acf
    filas = {}
    for v in [*VARIABLES_F3, BINARIA]:
        d = obs[['codigo', v]].dropna()
        s = d[v].astype(float)
        grupos = [g[v].to_numpy(float) for _, g in d.groupby('codigo')]
        p_adf, p_kpss = _p_estacionariedad(s.to_numpy())
        p = s.mean()
        filas[v] = {
            'n': len(s), 'media': p, 'mediana': s.median(), 'sd': s.std(), 'min': s.min(), 'max': s.max(),
            'RIC': s.quantile(.75) - s.quantile(.25), 'asimetria': stats.skew(s), 'curtosis': stats.kurtosis(s),
            'CV': s.std() / p if v != BINARIA else np.nan,
            'IVC': 4 * p * (1 - p) if v == BINARIA else np.nan,           # variación cualitativa con 2 categorías
            'indice_dispersion': s.var() / p, 'pct_ceros': 100 * (s == 0).mean(),
            'p_levene_jugadores': stats.levene(*grupos, center='median').pvalue if v != BINARIA else np.nan,
            'p_kruskal_jugadores': stats.kruskal(*grupos).pvalue,
            'p_ADF': p_adf, 'p_KPSS': p_kpss, 'estacionaria': (p_adf < ALFA) and (p_kpss >= ALFA),
            'acf1_torneo': acf(s, nlags=1)[1],
            'p_ljung_box_10': float(acorr_ljungbox(s, lags=[10])['lb_pvalue'].iloc[0]),
        }
    return pd.DataFrame(filas)


def acf1_por_jugador(obs):
    """ACF de rezago 1 de cada jugador sobre su propia secuencia de puntos (D51): mediana y % significativa al 5 %."""
    filas = []
    for v in [*VARIABLES_F3, BINARIA]:
        r = []
        for _, g in obs[['codigo', v]].dropna().groupby('codigo'):
            s = g[v].to_numpy(float)
            if len(s) >= MIN_OBS_JUGADOR and s.std() > 0:
                r.append((np.corrcoef(s[:-1], s[1:])[0, 1], 1.96 / np.sqrt(len(s))))
        r = np.array(r)
        filas.append({'variable': v, 'jugadores': len(r), 'acf1_mediana': np.nanmedian(r[:, 0]),
                      'pct_significativa': 100 * np.mean(np.abs(r[:, 0]) > r[:, 1])})
    return pd.DataFrame(filas).set_index('variable')


def graficar_univariado(obs, variables):
    """Por variable: histograma + KDE, violín, ECDF, Q-Q normal, Q-Q exponencial, secuencia + media móvil y ACF."""
    from statsmodels.graphics.tsaplots import plot_acf
    titulos = ['Histograma + KDE', 'Violín', 'ECDF', 'Q-Q normal', 'Q-Q exponencial',
               f'Secuencia del torneo (media móvil {VENTANA_MOVIL})', 'ACF (orden del torneo)']
    fig, ejes = plt.subplots(len(variables), 7, figsize=(26, 3.1 * len(variables)))
    for fila, v in zip(np.atleast_2d(ejes), variables):
        s = obs[v].dropna().astype(float)
        movil = s.rolling(VENTANA_MOVIL, min_periods=VENTANA_MOVIL // 2).mean()
        if v == BINARIA:
            fila[0].bar(['defensor', 'atacante'], [(s == 0).mean(), (s == 1).mean()], color=COLOR, width=0.5)
            fila[5].plot(np.arange(len(s)), movil, color=COLOR)
            for ax in fila[[1, 2, 3, 4]]:
                ax.axis('off')
            titulos_fila = ['Proporción', '', '', '', '', f'Tasa móvil de atacante ({VENTANA_MOVIL})', titulos[6]]
        else:
            fila[0].hist(s, bins=30, density=True, color=COLOR, alpha=0.6, edgecolor='white')
            if s.nunique() > 2:
                xs = np.linspace(s.min(), s.max(), 200)
                fila[0].plot(xs, stats.gaussian_kde(s)(xs), color=COLOR)
            fila[1].violinplot(s, orientation='horizontal', showmedians=True)
            fila[2].ecdf(s, color=COLOR)
            stats.probplot(s, dist='norm', plot=fila[3])
            stats.probplot(s, dist='expon', plot=fila[4])
            for ax in fila[[3, 4]]:
                ax.get_lines()[0].set(color=COLOR, markersize=2)
                ax.get_lines()[1].set(color=TINTA)
            fila[5].plot(np.arange(len(s)), s.to_numpy(), color=GRIS, linewidth=0.5)
            fila[5].plot(np.arange(len(s)), movil.to_numpy(), color=COLOR, linewidth=1.5)
            titulos_fila = titulos
        plot_acf(s, lags=20, ax=fila[6], color=COLOR, vlines_kwargs={'colors': COLOR}, title='')
        for ax, t in zip(fila, titulos_fila):
            if ax.axison:
                ax.set_title('')                          # quita el título centrado de probplot
                ax.set_title(f'{v} — {t}' if t else '', fontsize=9, loc='left')
                _estilo(ax)
        fila[1].set_yticks([])
    fig.tight_layout()
    return fig


def tasas_por_jugador(obs):
    """Proporciones por jugador: victoria, atacante y las partes Bernoulli de D53."""
    g = obs.groupby(['codigo', 'nombre'])
    t = pd.DataFrame({
        'puntos': g.size(), 'tasa_victoria': g['gano_punto'].mean(), 'tasa_atacante': g[BINARIA].mean(),
        'p_100_su_lado': g['pct_en_su_lado'].agg(lambda s: (s.dropna() == 100).mean()),
        'p_0_rafagas': g['rafagas_por_min'].agg(lambda s: (s.dropna() == 0).mean()),
    }).reset_index()
    return t.sort_values('tasa_atacante', ascending=False).reset_index(drop=True)


def homogeneidad_bernoulli(obs):
    """Chi² jugador × valor: ¿una sola proporción sirve para todos los jugadores?"""
    filas = {}
    for comp in ['p_100_su_lado', 'p_0_rafagas', 'atacante']:
        d = valores_componente(obs, comp)
        prop = d.groupby('codigo')['x'].mean()
        chi2, p, gl, _ = stats.chi2_contingency(pd.crosstab(d['codigo'], d['x']))
        filas[comp] = {'p_global': d['x'].mean(), 'p_min_jugador': prop.min(), 'p_max_jugador': prop.max(),
                       'chi2': chi2, 'gl': gl, 'p_valor': p}
    return pd.DataFrame(filas).T


def graficar_homogeneidad(obs, tasas):
    """Caja por jugador (ordenados por mediana) de las 7 variables y tasa de atacante por jugador."""
    fig, ejes = plt.subplots(2, 4, figsize=(24, 8))
    for ax, v in zip(ejes.flat, VARIABLES_F3):
        d = obs[['codigo', v]].dropna()
        orden = d.groupby('codigo')[v].median().sort_values().index
        ax.boxplot([d.loc[d['codigo'] == c, v] for c in orden], showfliers=False,
                   patch_artist=True, boxprops={'facecolor': COLOR, 'alpha': 0.5}, medianprops={'color': TINTA})
        ax.set(title=f'{v} por jugador (ordenados por mediana)', xticks=[], xlabel='jugador')
        _estilo(ax)
    ax = ejes.flat[-1]
    ax.bar(range(len(tasas)), tasas['tasa_atacante'], color=COLOR)
    ax.axhline(obs[BINARIA].mean(), color=TINTA, linestyle='--', linewidth=1)
    ax.set(title='Tasa de atacante por jugador (línea: global)', xticks=[], xlabel='jugador', ylim=(0, 1))
    _estilo(ax)
    fig.tight_layout()
    return fig


def comparar_grupos(obs, grupo):
    """Media y mediana de cada variable por grupo (0/1) y Mann-Whitney (3.2)."""
    filas = {}
    for v in [c for c in [*VARIABLES_F3, BINARIA] if c != grupo]:
        d = obs[[grupo, v]].dropna()
        a, b = d.loc[d[grupo] == 0, v], d.loc[d[grupo] == 1, v]
        filas[v] = {f'media_{grupo}=0': a.mean(), f'media_{grupo}=1': b.mean(),
                    f'mediana_{grupo}=0': a.median(), f'mediana_{grupo}=1': b.median(),
                    'p_mann_whitney': stats.mannwhitneyu(a, b).pvalue}
    return pd.DataFrame(filas).T


def pares_en_el_punto(obs, v):
    """Pares del mismo punto: compañeros (LEFT, RIGHT de un equipo) y rivales del mismo lado (equipo 1, equipo 2)."""
    d = obs[['partido', 'punto', 'equipo', 'lado', v]].dropna()
    comp = d.pivot_table(index=['partido', 'punto', 'equipo'], columns='lado', values=v).dropna()
    riv = d.pivot_table(index=['partido', 'punto', 'lado'], columns='equipo', values=v).dropna()
    return comp, riv


def correlacion_en_el_punto(obs):
    """Spearman entre compañeros y entre rivales del mismo lado, dentro del mismo punto."""
    filas = {}
    for v in [*VARIABLES_F3, BINARIA]:
        comp, riv = pares_en_el_punto(obs, v)
        rc, pc = stats.spearmanr(comp['LEFT'], comp['RIGHT'])
        rr, pr = stats.spearmanr(riv[1], riv[2])
        filas[v] = {'rho_companeros': rc, 'p_companeros': pc, 'pares_companeros': len(comp),
                    'rho_rivales_mismo_lado': rr, 'p_rivales': pr, 'pares_rivales': len(riv)}
    return pd.DataFrame(filas).T


def graficar_bivariado(obs):
    """Violín por resultado del punto, partido por estilo, y conteo estilo × resultado."""
    import seaborn as sns
    d = obs.assign(estilo=obs[BINARIA].map({0: 'defensor', 1: 'atacante'}),
                   resultado=obs['gano_punto'].map({0: 'pierde', 1: 'gana'}))
    fig, ejes = plt.subplots(2, 4, figsize=(24, 9))
    for i, (ax, v) in enumerate(zip(ejes.flat, VARIABLES_F3)):
        sns.violinplot(data=d, x='resultado', y=v, hue='estilo', split=True, inner='quart', ax=ax,
                       order=['pierde', 'gana'], hue_order=['defensor', 'atacante'], palette=[GRIS, COLOR],
                       legend=(i == 0), cut=0)
        ax.set(title=f'{v} por resultado y estilo', xlabel='')
        _estilo(ax)
    ax = ejes.flat[-1]
    sns.countplot(data=d.dropna(subset=['estilo']), x='resultado', hue='estilo', order=['pierde', 'gana'],
                  hue_order=['defensor', 'atacante'], palette=[GRIS, COLOR], ax=ax)
    ax.set(title='Estilo × resultado (jugador × punto)', xlabel='', ylabel='jugador × punto')
    _estilo(ax)
    fig.tight_layout()
    return fig


def graficar_companeros(obs):
    """Valor del jugador de la izquierda contra el de la derecha, en el mismo punto y equipo."""
    fig, ejes = plt.subplots(2, 4, figsize=(24, 10))
    for ax, v in zip(ejes.flat, VARIABLES_F3):
        comp, _ = pares_en_el_punto(obs, v)
        ax.scatter(comp['LEFT'], comp['RIGHT'], s=10, alpha=0.5, color=COLOR)
        rho = stats.spearmanr(comp['LEFT'], comp['RIGHT'])[0]
        ax.set(title=f'{v}: compañeros (Spearman {rho:.2f})', xlabel='jugador LEFT', ylabel='jugador RIGHT')
        _estilo(ax, 'both')
    comp, _ = pares_en_el_punto(obs, BINARIA)
    tabla = pd.crosstab(comp['LEFT'].map({0: 'defensor', 1: 'atacante'}), comp['RIGHT'].map({0: 'defensor', 1: 'atacante'}))
    ax = ejes.flat[-1]
    ax.imshow(tabla.to_numpy(), cmap='Blues')
    for (i, j), n in np.ndenumerate(tabla.to_numpy()):
        ax.text(j, i, n, ha='center', va='center', fontsize=13)
    ax.set(title='Estilo de los compañeros', xticks=[0, 1], yticks=[0, 1], xticklabels=tabla.columns,
           yticklabels=tabla.index, xlabel='jugador RIGHT', ylabel='jugador LEFT')
    fig.tight_layout()
    return fig


def graficar_joint(obs, x, y):
    import seaborn as sns
    d = obs.assign(estilo=obs[BINARIA].map({0: 'defensor', 1: 'atacante'})).dropna(subset=[x, y, 'estilo'])
    g = sns.jointplot(data=d, x=x, y=y, hue='estilo', palette={'defensor': GRIS, 'atacante': COLOR},
                      height=6, alpha=0.6, s=15)
    g.figure.suptitle(f'{x} contra {y} (Spearman {stats.spearmanr(d[x], d[y])[0]:.2f})', y=1.02)
    return g.figure


def graficar_multivariado(obs):
    """Matrices de Pearson y Spearman y correlación de cada variable con el resultado y con el estilo (3.3)."""
    cols = [*VARIABLES_F3, BINARIA, 'gano_punto']
    d = obs[cols].astype(float)
    fig, ejes = plt.subplots(1, 3, figsize=(24, 7.5), gridspec_kw={'width_ratios': [1, 1, 0.7]}, layout='constrained')
    for ax, metodo in zip(ejes[:2], ['pearson', 'spearman']):
        m = d.corr(method=metodo)
        im = ax.imshow(m.to_numpy(), cmap='RdBu_r', vmin=-1, vmax=1)
        for (i, j), r in np.ndenumerate(m.to_numpy()):
            ax.text(j, i, f'{r:.2f}', ha='center', va='center', fontsize=8)
        ax.set(xticks=range(len(cols)), yticks=range(len(cols)), title=f'Correlación de {metodo.capitalize()}')
        ax.set_xticklabels(cols, rotation=45, ha='right')
        ax.set_yticklabels(cols)
    fig.colorbar(im, ax=ejes[1], shrink=0.8)
    sp = d.corr(method='spearman')
    vs = [*VARIABLES_F3]
    yy = np.arange(len(vs))
    ejes[2].barh(yy - 0.2, sp.loc[vs, 'gano_punto'], height=0.4, color=COLOR, label='con gano_punto')
    ejes[2].barh(yy + 0.2, sp.loc[vs, BINARIA], height=0.4, color=GRIS, label='con atacante')
    ejes[2].set(yticks=yy, yticklabels=vs, title='Spearman con el resultado y con el estilo', xlim=(-1, 1))
    ejes[2].axvline(0, color=TINTA, linewidth=0.8)
    ejes[2].legend(fontsize=8)
    _estilo(ejes[2], 'x')
    return fig


def graficar_pairplot(obs):
    import seaborn as sns
    d = obs.assign(estilo=obs[BINARIA].map({0: 'defensor', 1: 'atacante'})).dropna(subset=[*VARIABLES_F3, 'estilo'])
    g = sns.pairplot(d[[*VARIABLES_F3, 'estilo']], hue='estilo', corner=True, palette={'defensor': GRIS, 'atacante': COLOR},
                     plot_kws={'s': 6, 'alpha': 0.4}, diag_kind='kde', height=1.9)
    return g.figure


# ══ 3. Ajuste de distribuciones (3.4) ══════════════════════════════════════════
CANDIDATAS_CONTINUAS = ['normal', 'uniforme', 'triangular', 'exponencial', 'gamma', 'erlang', 'chi2', 'lognormal',
                        'weibull', 'beta']
CANDIDATAS_CONTEO = ['poisson', 'binomial_negativa']
SCIPY = {'normal': 'norm', 'uniforme': 'uniform', 'triangular': 'triang', 'exponencial': 'expon', 'gamma': 'gamma',
         'erlang': 'gamma', 'chi2': 'chi2', 'lognormal': 'lognorm', 'weibull': 'weibull_min', 'beta': 'beta',
         'poisson': 'poisson', 'binomial_negativa': 'nbinom', 'bernoulli': 'bernoulli'}


class Truncada:
    """Distribución continua truncada y renormalizada en [lo, hi] (D56). Imita lo que usa el resto del módulo."""

    def __init__(self, base, lo, hi):
        self.base, self.lo, self.hi = base, lo, hi
        self.f_lo, self.f_hi = base.cdf(lo), base.cdf(hi)
        self.z = self.f_hi - self.f_lo

    def pdf(self, x):
        x = np.asarray(x, float)
        return np.where((x >= self.lo) & (x <= self.hi), self.base.pdf(x) / self.z, 0.0)

    def logpdf(self, x):
        with np.errstate(divide='ignore'):
            return np.log(self.pdf(x))

    def cdf(self, x):
        return np.clip((self.base.cdf(x) - self.f_lo) / self.z, 0, 1)

    def ppf(self, q):
        return self.base.ppf(self.f_lo + np.asarray(q) * self.z)

    def rvs(self, size=None, random_state=None):
        return self.ppf(np.random.default_rng(random_state).uniform(size=size))

    def mean(self):
        return integrate.quad(lambda t: t * self.base.pdf(t) / self.z, self.lo, self.hi, limit=200)[0]


def _limites_beta(x, lo, hi):
    """Corrección de Smithson y Verkuilen: lleva los 0 y 1 exactos al interior de (lo, hi) (D54)."""
    n = len(x)
    y = (x - lo) / (hi - lo)
    return lo + (y * (n - 1) + 0.5) / n * (hi - lo)


def ajustar_familia(familia, x, limites=(0, None)):
    """MLE de una familia (D54). Devuelve (distribución, texto de parámetros, k, kwargs de scipy, datos evaluados)."""
    x = np.asarray(x, float)
    lo, hi = limites
    xe = x
    if familia == 'normal':
        mu, s = stats.norm.fit(x)
        kw, txt = {'loc': mu, 'scale': s}, f'mu={mu:.4g}, sigma={s:.4g}'
    elif familia == 'uniforme':
        kw, txt = {'loc': x.min(), 'scale': x.max() - x.min()}, f'a={x.min():.4g}, b={x.max():.4g}'
    elif familia == 'triangular':
        eps = 1e-4 * (x.max() - x.min())      # extremos apenas por fuera para que min y max tengan densidad > 0
        a, b = x.min() - eps, x.max() + eps
        nll = lambda c: -stats.triang.logpdf(x, c, loc=a, scale=b - a).sum()
        c = optimize.minimize_scalar(nll, bounds=(1e-6, 1 - 1e-6), method='bounded').x
        kw, txt = {'c': c, 'loc': a, 'scale': b - a}, f'a={a:.4g}, moda={a + c * (b - a):.4g}, b={b:.4g}'
    elif familia == 'exponencial':
        kw, txt = {'loc': 0, 'scale': x.mean()}, f'lambda={1 / x.mean():.4g}'
    elif familia == 'gamma':
        k, _, th = stats.gamma.fit(x, floc=0)
        kw, txt = {'a': k, 'loc': 0, 'scale': th}, f'forma_k={k:.4g}, escala_theta={th:.4g}'
    elif familia == 'erlang':
        k = max(1, int(round(stats.gamma.fit(x, floc=0)[0])))
        kw, txt = {'a': k, 'loc': 0, 'scale': x.mean() / k}, f'forma_k (entero)={k}, escala_theta={x.mean() / k:.4g}'
    elif familia == 'chi2':
        gl = stats.chi2.fit(x, floc=0, fscale=1)[0]
        kw, txt = {'df': gl}, f'gl={gl:.4g}'
    elif familia == 'lognormal':
        s, _, esc = stats.lognorm.fit(x, floc=0)
        kw, txt = {'s': s, 'loc': 0, 'scale': esc}, f'mu_log={np.log(esc):.4g}, sigma_log={s:.4g}'
    elif familia == 'weibull':
        c, _, esc = stats.weibull_min.fit(x, floc=0)
        kw, txt = {'c': c, 'loc': 0, 'scale': esc}, f'forma_c={c:.4g}, escala={esc:.4g}'
    elif familia == 'beta':
        xe = _limites_beta(x, lo, hi)
        a, b, _, _ = stats.beta.fit((xe - lo) / (hi - lo), floc=0, fscale=1)
        kw, txt = {'a': a, 'b': b, 'loc': lo, 'scale': hi - lo}, f'alfa={a:.4g}, beta={b:.4g}, a={lo}, b={hi}'
    elif familia == 'poisson':
        kw, txt = {'mu': x.mean()}, f'lambda={x.mean():.4g}'
    elif familia == 'binomial_negativa':
        m, v = x.mean(), x.var()
        p0 = min(max(m / v, 0.01), 0.99) if v > 0 else 0.5
        z0 = [np.log(m * p0 / (1 - p0)), np.log(p0 / (1 - p0))]
        nll = lambda z: -stats.nbinom.logpmf(x, np.exp(z[0]), 1 / (1 + np.exp(-z[1]))).sum()
        z = optimize.minimize(nll, z0, method='Nelder-Mead', options={'xatol': 1e-8, 'fatol': 1e-8, 'maxiter': 4000}).x
        r, p = np.exp(z[0]), 1 / (1 + np.exp(-z[1]))
        kw, txt = {'n': r, 'p': p}, f'r={r:.4g}, p={p:.4g}'
    elif familia == 'bernoulli':
        kw, txt = {'p': x.mean()}, f'p={x.mean():.4g}'
    else:
        raise ValueError(familia)
    k = {'triangular': 3, 'exponencial': 1, 'chi2': 1, 'poisson': 1, 'bernoulli': 1}.get(familia, 2)
    return getattr(stats, SCIPY[familia])(**kw), txt, k, kw, xe


def _wilson(x):
    n, p, z = len(x), np.mean(x), 1.96
    c = (p + z ** 2 / (2 * n)) / (1 + z ** 2 / n)
    h = z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / (1 + z ** 2 / n)
    return c - h, c + h


def prueba_conteo(x, dist, k):
    """Chi² de bondad de ajuste para conteos: clases desde 0 que se juntan hasta esperar ≥ 5 (la última incluye la cola)."""
    x = np.asarray(x, int)
    n, obs_, esp_, o, e = len(x), [], [], 0, 0.0
    for v in range(x.max() + 1):
        o += (x == v).sum()
        e += n * dist.pmf(v)
        if e >= 5:
            obs_.append(o), esp_.append(e)
            o, e = 0, 0.0
    e += n * dist.sf(x.max())                       # lo que queda y la cola superior van a la última clase
    if esp_:
        obs_[-1] += o
        esp_[-1] += e
    else:
        obs_, esp_ = [o], [e]
    gl = len(obs_) - 1 - k
    est = sum((a - b) ** 2 / b for a, b in zip(obs_, esp_))
    return est, gl, stats.chi2.sf(est, gl) if gl > 0 else np.nan


def evaluar(x, familia, limites, tipo):
    """Ajusta una candidata y calcula log-verosimilitud, AIC, BIC y la prueba de bondad (D52)."""
    dist, txt, k, kw, xe = ajustar_familia(familia, x, limites)
    n = len(x)
    if tipo == 'bernoulli':
        ll = stats.binom.logpmf(np.asarray(x, int), 1, kw['p']).sum()
        lo_, hi_ = _wilson(x)
        prueba, est, p = f'IC95 % Wilson [{lo_:.3f}; {hi_:.3f}]', np.nan, np.nan
    elif tipo == 'conteo':
        ll = dist.logpmf(np.asarray(x, int)).sum()
        est, gl, p = prueba_conteo(x, dist, k)
        prueba = f'chi² (gl={gl})'
    else:
        ll = dist.logpdf(xe).sum()
        est, p = stats.kstest(xe, dist.cdf)
        prueba = 'KS'
    fila = {'distribucion': familia, 'parametros_MLE': txt, 'k': k, 'log_verosimilitud': ll,
            'AIC': 2 * k - 2 * ll, 'BIC': k * np.log(n) - 2 * ll, 'prueba': prueba, 'estadistico': est, 'p_valor': p}
    return fila, dist, kw


def elegir(tabla):
    """D52: menor AIC entre las no rechazadas (α = 0,05); si todas se rechazan, la de menor AIC y bootstrap empírico."""
    if tabla['p_valor'].isna().all():       # Bernoulli: no tiene prueba de bondad
        return tabla.iloc[0], 'paramétrica'
    ok = tabla[tabla['p_valor'] >= ALFA]
    if len(ok):
        return ok.iloc[0], 'paramétrica'
    return tabla.iloc[0], 'bootstrap empírico (todas rechazadas)'


def _sin_ceros(x):
    """D54: los ceros técnicos pasan a la mitad del menor valor positivo (para las familias con soporte (0, ∞))."""
    x = np.asarray(x, float).copy()
    if (x == 0).any():
        x[x == 0] = x[x > 0].min() / 2
    return x


def masa_fuera(dist, limites):
    """Probabilidad que la distribución pone fuera de los límites de la variable."""
    lo, hi = limites
    return float(dist.cdf(lo) + (dist.sf(hi) if hi is not None else 0.0))


def ajustar_truncada(familia, x, limites, kw0):
    """D56: MLE de la familia truncada a los límites de la variable, partiendo de los parámetros sin truncar.

    Solo se estiman los parámetros libres de la familia (el origen fijo en 0 y la forma entera de la Erlang se
    mantienen). Devuelve (distribución truncada, kwargs de la base, log-verosimilitud).
    """
    lo, hi = limites[0], limites[1] if limites[1] is not None else np.inf
    nombre = SCIPY[familia]
    libres = [k for k in kw0 if not (k == 'loc' and familia != 'normal') and not (k == 'a' and familia == 'erlang')]
    positivo = [k != 'loc' for k in libres]
    z0 = [np.log(kw0[k]) if pos else kw0[k] for k, pos in zip(libres, positivo)]

    def armar(z):
        return {**kw0, **{k: (np.exp(v) if pos else v) for k, v, pos in zip(libres, z, positivo)}}

    def nll(z):
        t = Truncada(getattr(stats, nombre)(**armar(z)), lo, hi)
        ll = t.logpdf(x).sum()
        return -ll if np.isfinite(ll) else 1e12

    z = optimize.minimize(nll, z0, method='Nelder-Mead', options={'xatol': 1e-8, 'fatol': 1e-8, 'maxiter': 4000}).x
    kw = armar(z)
    t = Truncada(getattr(stats, nombre)(**kw), lo, hi)
    return t, kw, float(t.logpdf(x).sum())


def ajustar_componente(x, comp):
    """Todas las candidatas de un componente, ordenadas por AIC, con la elegida (D52) y su versión truncada (D56)."""
    _, tipo, limites, _ = COMPONENTES[comp]
    x = np.asarray(x, float)
    if tipo == 'continua':
        x = _sin_ceros(x)
        familias = [f for f in CANDIDATAS_CONTINUAS if f != 'beta' or limites[1] is not None]
    else:
        familias = CANDIDATAS_CONTEO if tipo == 'conteo' else ['bernoulli']
    filas, dists, kws = [], {}, {}
    for f in familias:
        fila, dists[f], kws[f] = evaluar(x, f, limites, tipo)
        filas.append(fila)
    tabla = pd.DataFrame(filas).sort_values('AIC').reset_index(drop=True)
    mejor, decision = elegir(tabla)
    f = mejor['distribucion']
    fuera = masa_fuera(dists[f], limites) if tipo == 'continua' else 0.0
    final, kw_final, p_final, truncada = dists[f], kws[f], mejor['p_valor'], False
    if tipo == 'continua' and decision == 'paramétrica' and fuera > 1e-3:      # D56
        final, kw_final, _ = ajustar_truncada(f, x, limites, kws[f])
        p_final, truncada = stats.kstest(x, final.cdf).pvalue, True
        if p_final < ALFA:          # la versión truncada también pasa por D52: si se rechaza, bootstrap empírico
            decision = 'bootstrap empírico (truncada rechazada)'
    return {'tipo': tipo, 'limites': limites, 'x': x, 'tabla': tabla, 'dists': dists, 'kwargs': kws,
            'elegida': f, 'decision': decision, 'final': final, 'kwargs_final': kw_final, 'p_final': p_final,
            'truncada': truncada, 'masa_fuera': fuera}


def ajustar_todo(obs):
    """Ajuste global de los 11 componentes (3.4)."""
    return {comp: ajustar_componente(valores_componente(obs, comp)['x'], comp) for comp in COMPONENTES}


def graficar_ajustes(resultados, componentes, top=4):
    """Por componente: densidad (o masa) sobre el histograma, ECDF contra CDF, Q-Q y P-P de la elegida.

    En densidad y CDF se dibujan las `top` candidatas de menor AIC; la elegida va en trazo grueso.
    """
    fig, ejes = plt.subplots(len(componentes), 4, figsize=(24, 4.3 * len(componentes)))
    for fila, comp in zip(np.atleast_2d(ejes), componentes):
        r = resultados[comp]
        x, elegida = r['x'], r['elegida']
        nombres = list(dict.fromkeys([elegida, *r['tabla']['distribucion'].head(top)]))
        colores = dict(zip(nombres, [COLOR, '#e07b39', '#3aa676', '#8e6bbf', '#c44e52']))
        if r['tipo'] == 'conteo':
            vals = np.arange(0, int(np.quantile(x, 0.99)) + 1)
            fila[0].bar(vals, [(x == v).mean() for v in vals], color=GRIS, width=0.8, label='observado')
            for f in nombres:
                fila[0].plot(vals, r['dists'][f].pmf(vals), 'o-', ms=3, color=colores[f], lw=2.5 if f == elegida else 1, label=f)
        else:
            fila[0].hist(x, bins=40, density=True, color=GRIS, alpha=0.7, edgecolor='white', label='observado')
            xs = np.linspace(x.min(), x.max(), 300)
            for f in nombres:
                fila[0].plot(xs, r['dists'][f].pdf(xs), color=colores[f], lw=2.5 if f == elegida else 1, label=f)
            if r['final'] is not r['dists'][elegida]:
                fila[0].plot(xs, r['final'].pdf(xs), '--', color=COLOR, lw=1.5, label=f'{elegida} truncada')
        fila[1].ecdf(x, color=GRIS, label='observado')
        xs = np.linspace(x.min(), x.max(), 300)
        for f in nombres:
            fila[1].plot(xs, r['dists'][f].cdf(xs), color=colores[f], lw=2.5 if f == elegida else 1, label=f)
        q = (np.arange(1, len(x) + 1) - 0.5) / len(x)
        xo = np.sort(x)
        fila[2].scatter(r['final'].ppf(q), xo, s=6, color=COLOR)
        fila[3].scatter(q, r['final'].cdf(xo), s=6, color=COLOR)
        lim = [min(xo.min(), np.nanmin(r['final'].ppf(q))), max(xo.max(), np.nanmax(r['final'].ppf(q)))]
        fila[2].plot(lim, lim, color=TINTA, lw=0.8)
        fila[3].plot([0, 1], [0, 1], color=TINTA, lw=0.8)
        tit = ['Densidad' if r['tipo'] != 'conteo' else 'Masa de probabilidad', 'ECDF y CDF',
               f'Q-Q ({elegida})', f'P-P ({elegida})']
        for ax, t in zip(fila, tit):
            ax.set_title(f'{comp} — {t}', fontsize=10, loc='left')
            _estilo(ax, 'both')
        fila[0].legend(fontsize=7)
        fila[1].legend(fontsize=7)
        fila[2].set(xlabel='cuantil teórico', ylabel='cuantil observado')
        fila[3].set(xlabel='probabilidad empírica', ylabel='probabilidad teórica')
    fig.tight_layout()
    return fig


# ══ 4. Por jugador y bootstrap (3.5) ═══════════════════════════════════════════
def _ic_bootstrap(x, rng, b=B_BOOT):
    """IC 95 % de la media y del percentil 90 remuestreando los datos con reemplazo."""
    x = np.asarray(x, float)
    m = x[rng.integers(0, len(x), size=(b, len(x)))]
    return np.quantile(m.mean(axis=1), [0.025, 0.975]), np.quantile(np.quantile(m, 0.9, axis=1), [0.025, 0.975])


def _media(dist):
    return dist.mean() if isinstance(dist, Truncada) else float(dist.mean())


def bootstrap_vs_parametrica(resultados):
    """¿La distribución elegida (global) reproduce la media y el p90 del bootstrap empírico? (README, fase 3)."""
    rng = np.random.default_rng(SEMILLA)
    filas = {}
    for comp, r in resultados.items():
        if r['tipo'] == 'bernoulli':
            continue
        (m_lo, m_hi), (p_lo, p_hi) = _ic_bootstrap(r['x'], rng)
        mp, pp = _media(r['final']), float(r['final'].ppf(0.9))
        filas[comp] = {'distribucion': r['elegida'] + (' truncada' if isinstance(r['final'], Truncada) else ''),
                       'decision': r['decision'], 'media_obs': r['x'].mean(), 'media_IC95_boot': f'[{m_lo:.3f}; {m_hi:.3f}]',
                       'media_param': mp, 'media_en_IC': m_lo <= mp <= m_hi, 'p90_obs': np.quantile(r['x'], 0.9),
                       'p90_IC95_boot': f'[{p_lo:.3f}; {p_hi:.3f}]', 'p90_param': pp, 'p90_en_IC': p_lo <= pp <= p_hi}
    return pd.DataFrame(filas).T


def ajustar_por_jugador(obs, resultados):
    """D57: por jugador y componente, familia global con sus parámetros si la prueba no la rechaza; si la global fue
    rechazada o la del jugador se rechaza, bootstrap empírico del jugador; con < 10 observaciones, la global (D50).

    Devuelve una fila por jugador × componente con lo que necesita la fase 4 para sortear (familia de scipy y kwargs).
    """
    rng = np.random.default_rng(SEMILLA)
    nombres = obs.drop_duplicates('codigo').set_index('codigo')['nombre']
    filas = []
    for comp, r in resultados.items():
        if comp in SOLO_COMPARACION:
            continue
        d = valores_componente(obs, comp)
        familia, tipo, limites = r['elegida'], r['tipo'], r['limites']
        for codigo in nombres.index:
            x = d.loc[d['codigo'] == codigo, 'x'].to_numpy(float)
            fila = {'codigo': codigo, 'nombre': nombres[codigo], 'componente': comp, 'n': len(x), 'familia': familia,
                    'scipy': SCIPY[familia], 'kwargs': '', 'parametros': '', 'truncada': False, 'p_valor': np.nan,
                    'media_obs': x.mean() if len(x) else np.nan}
            if len(x) < MIN_OBS_JUGADOR:
                fila.update(metodo='global', kwargs=json.dumps(r['kwargs_final']),
                            parametros=_texto_kwargs(r['kwargs_final']), truncada=r['truncada'])
            elif r['decision'] != 'paramétrica':
                fila.update(metodo='bootstrap')
            else:
                xa = _sin_ceros(x) if tipo == 'continua' else x
                if tipo == 'continua' and (xa.std() == 0):
                    fila.update(metodo='bootstrap')
                else:
                    ev, final, kw = evaluar(xa, familia, limites, tipo)
                    p = ev['p_valor']
                    if r['truncada']:      # la familia global va truncada (D56): el jugador también
                        final, kw, _ = ajustar_truncada(familia, xa, limites, kw)
                        p = stats.kstest(xa, final.cdf).pvalue
                    rechazada = pd.notna(p) and p < ALFA
                    fila.update(metodo='bootstrap' if rechazada else 'parametrica', kwargs=json.dumps(kw),
                                parametros=_texto_kwargs(kw), p_valor=p, truncada=r['truncada'])
                    if tipo != 'bernoulli' and not rechazada:
                        (m_lo, m_hi), (p_lo, p_hi) = _ic_bootstrap(xa, rng)
                        mp, pp = _media(final), float(final.ppf(0.9))
                        fila.update(media_param=mp, media_en_IC=m_lo <= mp <= m_hi, p90_param=pp, p90_en_IC=p_lo <= pp <= p_hi)
            filas.append(fila)
    return pd.DataFrame(filas)


def _texto_kwargs(kw):
    return ', '.join(f'{k}={v:.4g}' for k, v in kw.items())


def resumen_por_jugador(por_jugador):
    """Por componente: jugadores con cada método (D57) y % de los paramétricos cuya media y p90 caen en su IC bootstrap."""
    t = pd.crosstab(por_jugador['componente'], por_jugador['metodo'])
    par = por_jugador[por_jugador['metodo'] == 'parametrica'].dropna(subset=['media_en_IC'])
    t['pct_media_en_IC'] = 100 * par.groupby('componente')['media_en_IC'].apply(lambda s: s.astype(bool).mean())
    t['pct_p90_en_IC'] = 100 * par.groupby('componente')['p90_en_IC'].apply(lambda s: s.astype(bool).mean())
    return t


def distribucion(fila, limites=None):
    """Reconstruye la distribución de una fila de distribuciones_global / distribuciones_jugador (para la fase 4)."""
    dist = getattr(stats, fila['scipy'])(**json.loads(fila['kwargs']))
    if fila.get('truncada', False):
        lo, hi = limites or COMPONENTES[fila['componente']][2]
        return Truncada(dist, lo, hi if hi is not None else np.inf)
    return dist


# ══ 5. Tabla final (3.6) ══════════════════════════════════════════════════════
def tabla_final(resultados, por_jugador):
    """Una fila por componente: elección global (D52), truncamiento (D56), segunda candidata y resumen por jugador (D57)."""
    metodos = por_jugador.pivot_table(index='componente', columns='metodo', values='codigo', aggfunc='count', fill_value=0)
    filas = []
    for comp, r in resultados.items():
        t = r['tabla'].set_index('distribucion')
        e = t.loc[r['elegida']]
        segunda = t.drop(index=r['elegida']).iloc[0] if len(t) > 1 else None
        fila = {'variable': COMPONENTES[comp][0], 'componente': comp, 'tipo': r['tipo'], 'definicion': COMPONENTES[comp][3],
                'n': len(r['x']), 'distribucion_elegida': r['elegida'], 'parametros_MLE': e['parametros_MLE'],
                'AIC': e['AIC'], 'prueba': e['prueba'], 'p_valor': e['p_valor'],
                'segunda': segunda.name if segunda is not None else '',
                'delta_AIC_segunda': segunda['AIC'] - e['AIC'] if segunda is not None else np.nan,
                'decision': r['decision'], 'pct_masa_fuera_de_limites': 100 * r['masa_fuera'], 'truncada': r['truncada'],
                'parametros_truncada': _texto_kwargs(r['kwargs_final']) if r['truncada'] else '',
                'p_valor_truncada': r['p_final'] if r['truncada'] else np.nan,
                'usar_en_fase4': comp not in SOLO_COMPARACION,
                'scipy': SCIPY[r['elegida']], 'kwargs': json.dumps(r['kwargs_final'])}
        for m in ['parametrica', 'bootstrap', 'global']:
            fila[f'jugadores_{m}'] = int(metodos.loc[comp, m]) if comp in metodos.index and m in metodos else 0
        filas.append(fila)
    return pd.DataFrame(filas)
