"""Variables del modelo: jugador × punto (1.3), correlación (1.4) y tabla equipo × punto (1.5)."""

import numpy as np
import pandas as pd

from .datos import DIST_RED_LINEA_SAQUE_M

LLAVE_JP = ['partido', 'punto', 'codigo']
MIN_DURACION_S = 2.0          # tracks más cortos: sin variables de video (D21)
UMBRAL_CORRELACION = 0.8      # |r de Spearman| desde el que dos variables «dicen lo mismo» (D24)

# Las 14 variables preliminares del jugador en 3 grupos (D21, D22, D15)
GRUPOS = {
    'nivel de juego': ['nivel_padel', 'tiempo_jugando_padel', 'experiencia_padel', 'practica_otra_raqueta'],
    'tipo de juego': ['atacante', 'dist_red_m', 'vel_media_mps', 'pct_quieto'],
    'atributos físicos': ['estatura_cm', 'imc', 'edad', 'estado_fisico', 'frecuencia_deporte', 'mujer'],
}
VARIABLES_JUGADOR = [v for grupo in GRUPOS.values() for v in grupo]
VARIABLES_VIDEO = GRUPOS['tipo de juego']
VARIABLES_FORMULARIO = [v for v in VARIABLES_JUGADOR if v not in VARIABLES_VIDEO]
MODELO_1 = ['nivel_padel', 'atacante', 'vel_media_mps', 'estatura_cm']     # las 4 del primer modelo (D31)


def variables_video(limpio, por_fondo=True):
    """Las 4 variables de tipo de juego por jugador × punto.

    por_fondo=True (D22): los umbrales y las medianas se calculan con los jugadores del mismo fondo.
    por_fondo=False: atacante con la línea de saque (6,95 m) y quieto con la mediana global; solo sirve
    para mostrar cuánto cambian las variables según el fondo.
    """
    v = limpio[[*LLAVE_JP, 'fondo', 'fps', 'dist_red_m', 'velocidad_mps']].copy()
    grupo = v['fondo'] if por_fondo else pd.Series('todos', index=v.index)
    linea = v.groupby(grupo)['dist_red_m'].transform('median') if por_fondo else DIST_RED_LINEA_SAQUE_M
    lento = v.groupby(grupo)['velocidad_mps'].transform('median')
    v['cerca_red'] = (v['dist_red_m'] < linea).astype(float).where(v['dist_red_m'].notna())
    v['quieto'] = (v['velocidad_mps'] < lento).astype(float).where(v['velocidad_mps'].notna())

    t = v.groupby([*LLAVE_JP, 'fondo']).agg(
        frames=('fps', 'size'), fps=('fps', 'first'), dist_red_m=('dist_red_m', 'mean'),
        vel_media_mps=('velocidad_mps', 'mean'), cerca_red=('cerca_red', 'mean'), pct_quieto=('quieto', 'mean'))
    t = t[t['frames'] / t['fps'] >= MIN_DURACION_S].copy()                     # D21
    t['atacante'] = (t['cerca_red'] >= 0.5).astype(float).where(t['cerca_red'].notna())
    t['pct_quieto'] = 100 * t['pct_quieto']
    if por_fondo:                                                               # respecto de su fondo (D22)
        for c in ['dist_red_m', 'vel_media_mps']:
            t[c] = t[c] - t.groupby('fondo')[c].transform('median')
    return t.reset_index()[[*LLAVE_JP, 'fondo', *VARIABLES_VIDEO]]


def variables_formulario(jugadores):
    """Las 10 variables fijas del jugador en escala numérica; se unen las categorías de una sola persona (D21)."""
    j = jugadores[['codigo', *VARIABLES_FORMULARIO]].copy()
    j['nivel_padel'] = j['nivel_padel'].clip(upper=2)                        # avanzado (1 jugador) se une a intermedio
    j['tiempo_jugando_padel'] = (j['tiempo_jugando_padel'] >= 1).astype(int)  # 0 = menos de 6 meses, 1 = 6 meses o más
    j['estado_fisico'] = (j['estado_fisico'] - 1).clip(lower=0)              # malo (1 jugador) se une a regular
    return j.astype({v: float for v in VARIABLES_FORMULARIO})


def efecto_fondo(jugador_punto, variables):
    """Jugadores que jugaron en los dos fondos: mediana de (su promedio en TOP − su promedio en BOTTOM)."""
    m = jugador_punto.groupby(['codigo', 'fondo'])[variables].mean().unstack('fondo')
    filas = []
    for v in variables:
        d = (m[(v, 'TOP')] - m[(v, 'BOTTOM')]).dropna()
        filas.append({'variable': v, 'jugadores': len(d), 'mediana TOP − BOTTOM': round(d.median(), 2),
                      'mayor en TOP': int((d > 0).sum())})
    return pd.DataFrame(filas).set_index('variable')


def pares_correlacionados(corr, umbral=UMBRAL_CORRELACION):
    """Pares de variables con |r| ≥ umbral, de mayor a menor."""
    arriba = corr.where(np.triu(np.ones(corr.shape, dtype=bool), k=1)).stack()
    return arriba[arriba.abs() >= umbral].sort_values(key=abs, ascending=False).rename('r de Spearman').to_frame()


def quitar_repetidas(pares):
    """De cada par repetido sale la variable que no está en el modelo 1 (D24)."""
    salen = []
    for a, b in pares.index:
        if (a in MODELO_1) == (b in MODELO_1):
            raise ValueError(f'{a} y {b}: ninguna o las dos están en el modelo 1; decidir con el equipo.')
        salen.append(b if a in MODELO_1 else a)
    return salen


def tabla_equipos(jugador_punto, puntos, variables):
    """Dos filas por punto, una por equipo: promedio de sus dos jugadores (propio_) y del rival (rival_) (D27).

    Solo puntos con los 4 jugadores identificados (D25) y con todas las variables de los 4 (D26).
    """
    completos = puntos.loc[puntos['equipos_completos'], ['partido', 'punto']]
    jp = jugador_punto.merge(completos, on=['partido', 'punto'])
    jp = jp.groupby(['partido', 'punto']).filter(lambda g: g[variables].notna().all().all())
    llave = ['partido', 'punto', 'equipo']
    equipo = jp.groupby(llave)[variables].mean()
    gano = jp.groupby(llave)['gano_punto'].first().rename('gano')
    rival = equipo.rename(index={1: 2, 2: 1}, level='equipo')    # a cada equipo le corresponde el otro
    tabla = pd.concat([gano, equipo.add_prefix('propio_'), rival.add_prefix('rival_')], axis=1)
    return tabla.reset_index().sort_values(llave).reset_index(drop=True)


def columnas(variables):
    """Las columnas del modelo para una lista de variables del jugador: la del equipo y la del rival."""
    return [f'{lado}_{v}' for v in variables for lado in ('propio', 'rival')]
