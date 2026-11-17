"""Variables de estilo de juego: jugador × punto (sección 1.9) y pareja × punto con filtros (sección 1.10).

Secciones:
1. Variables jugador × punto desde `video_limpio` (D21, D22, D31).
2. Variables pareja × punto, relativas al rival, y filtros sin mirar el resultado (D32, D33).
"""

import numpy as np
import pandas as pd

from .datos import DIST_RED_LINEA_SAQUE_M

LLAVE_JP = ['partido', 'punto', 'codigo']

# ══ 1. Jugador × punto (1.9) ═══════════════════════════════════════════════════
MIN_DURACION_S = 2.0          # tracks más cortos: todas sus variables NaN (D31)
MIN_POSE_COVERAGE = 50.0      # variables de pose solo si la cobertura es ≥ 50 % (D31)
RAFAGA_MIN_S = 0.5            # una ráfaga es un tramo continuo de ≥ 0,5 s sobre el umbral (D31)
CELDA_AREA_M = 0.5            # área cubierta: cuadros de 0,5 × 0,5 m pisados (D31)
GRUPOS_DIRECCION = {'arriba': ['up', 'up_left', 'up_right'], 'abajo': ['down', 'down_left', 'down_right'],
                    'lateral': ['left', 'right']}
VARIABLES_POSE = ['pct_juego_aereo', 'pct_raqueta_arriba', 'pct_raqueta_abajo', 'pct_raqueta_lateral',
                  'sd_angulo_codo', 'sd_angulo_hombro', 'sd_orientacion_raqueta', 'indice_golpe_p90',
                  'indice_golpe_p95', 'golpes_por_s']


def umbrales_movimiento(limpio):
    """Umbrales globales (iguales para todos los jugadores): quieto < mediana; ráfaga > p90 de la velocidad."""
    vel = limpio['velocidad_mps'].dropna()
    return {'quieto_mps': round(float(vel.median()), 3), 'rafaga_mps': round(float(vel.quantile(0.90)), 3)}


def _sd_circular(grados):
    """Desviación estándar circular en grados (−179° y 179° son vecinos)."""
    rad = np.deg2rad(grados.dropna().to_numpy())
    if len(rad) < 2:
        return np.nan
    r = np.hypot(np.cos(rad).mean(), np.sin(rad).mean())
    return float(np.rad2deg(np.sqrt(-2 * np.log(max(r, 1e-12)))))


def _rafagas(vel, fps, umbral):
    """Número de tramos continuos de al menos RAFAGA_MIN_S segundos con velocidad > umbral."""
    sobre = (vel > umbral).to_numpy()
    if not sobre.any():
        return 0
    cambios = np.diff(np.concatenate([[0], sobre.astype(int), [0]]))
    largos = np.flatnonzero(cambios == -1) - np.flatnonzero(cambios == 1)
    return int((largos >= RAFAGA_MIN_S * fps).sum())


def _area_rejilla(x, y):
    """m² de cuadros de CELDA_AREA_M de lado que el jugador pisó al menos una vez."""
    ok = x.notna() & y.notna()
    if not ok.any():
        return np.nan
    celdas = set(zip(np.floor(x[ok] / CELDA_AREA_M).astype(int), np.floor(y[ok] / CELDA_AREA_M).astype(int)))
    return len(celdas) * CELDA_AREA_M ** 2


def _pct(serie_bool):
    s = serie_bool.dropna()
    return 100 * float(s.astype(float).mean()) if len(s) else np.nan


def variables_jugador(limpio, umbrales):
    """Una fila por jugador × punto con las variables de estilo de juego (1.9)."""
    filas = []
    for (p, q, c), g in limpio.groupby(LLAVE_JP, sort=True):
        fps = float(g['fps'].iloc[0])
        r = {'partido': p, 'punto': q, 'codigo': c, 'duracion_track_s': round(len(g) / fps, 2)}
        if r['duracion_track_s'] < MIN_DURACION_S:
            filas.append(r)
            continue
        d = g['dist_red_m']
        con_y = d.notna()
        r['pct_zona_atacante'] = _pct(d[con_y] < DIST_RED_LINEA_SAQUE_M) if con_y.any() else np.nan
        r['atacante'] = float(r['pct_zona_atacante'] >= 50) if pd.notna(r['pct_zona_atacante']) else np.nan
        for sec in ['red', 'transicion', 'fondo']:
            r[f'pct_{sec}'] = _pct(g.loc[con_y, 'seccion'] == sec) if con_y.any() else np.nan
        r['prof_media_m'] = d.mean()
        r['prof_sd_m'] = d.std()
        r['pct_en_su_lado'] = _pct(g['en_su_lado'].astype('float'))
        r['area_m2'] = _area_rejilla(g['x_m'], g['y_m'])
        vel = g['velocidad_mps']
        r['distancia_m'] = g['desplazamiento_m'].sum(min_count=1)
        r['pct_quieto'] = _pct((vel < umbrales['quieto_mps']).where(vel.notna()))
        r['rafagas_por_min'] = _rafagas(vel, fps, umbrales['rafaga_mps']) / (r['duracion_track_s'] / 60) if vel.notna().any() else np.nan
        r['vel_p90'] = vel.quantile(0.90)
        r['pose_coverage'] = 100 * float(g['pose_disponible'].mean())
        if r['pose_coverage'] >= MIN_POSE_COVERAGE:
            r['pct_juego_aereo'] = _pct(g['juego_aereo'].astype('float'))
            direccion = g['direccion_raqueta'].dropna()
            for grupo, valores in GRUPOS_DIRECCION.items():
                r[f'pct_raqueta_{grupo}'] = 100 * float(direccion.isin(valores).mean()) if len(direccion) else np.nan
            r['sd_angulo_codo'] = g['angulo_codo'].std()
            r['sd_angulo_hombro'] = g['angulo_hombro'].std()
            r['sd_orientacion_raqueta'] = _sd_circular(g['orientacion_raqueta'])
            r['indice_golpe_p90'] = g['indice_vel_golpe'].quantile(0.90)
            r['indice_golpe_p95'] = g['indice_vel_golpe'].quantile(0.95)
            r['golpes_por_s'] = int(g['golpe'].sum()) / r['duracion_track_s']
        filas.append(r)
    out = pd.DataFrame(filas)
    for c in VARIABLES_POSE:
        if c not in out:
            out[c] = np.nan
    return out.round(4)


def estilo_jugador(var_jugador):
    """Estilo fijo del jugador: % de sus puntos como atacante y atacante si es ≥ 50 % (D21)."""
    e = var_jugador.dropna(subset=['atacante']).groupby('codigo')['atacante'].agg(['mean', 'size'])
    return pd.DataFrame({'codigo': e.index, 'puntos_con_estilo': e['size'].to_numpy(),
                         'pct_puntos_atacante': (100 * e['mean']).round(1).to_numpy(),
                         'estilo_atacante': (e['mean'] >= 0.5).astype(int).to_numpy()})


# ══ 2. Pareja × punto y filtros (1.10) ═════════════════════════════════════════
VARIABLES_JUGADOR_PAREJA = ['atacante', 'pct_zona_atacante', 'pct_red', 'pct_transicion', 'pct_fondo', 'prof_media_m',
                            'prof_sd_m', 'pct_en_su_lado', 'area_m2', 'distancia_m', 'pct_quieto', 'rafagas_por_min',
                            'vel_p90', 'pose_coverage', *VARIABLES_POSE]
VARIABLES_FIJAS = ['estatura_cm', 'imc', 'edad', 'mujer', 'zurdo', 'nivel_padel', 'tiempo_jugando_padel',
                   'experiencia_padel', 'practica_otra_raqueta', 'estado_fisico', 'frecuencia_deporte',
                   'lesion_ultimo_anio', 'en_lado_mano']
VARIABLES_CON_RAZON = ['vel_p90', 'golpes_por_s', 'area_m2']      # razón mín/máx además de suma y |dif| (D32)
UMBRAL_FALTANTES = 0.20       # filtros sin mirar el resultado (D33)
UMBRAL_VALOR_DOMINANTE = 0.95
UMBRAL_CORRELACION = 0.85     # Pearson, en valor absoluto


def _cohesion_y_sincronia(limpio):
    """Por pareja × punto: media y desviación de la distancia al compañero y correlaciones de velocidad y profundidad."""
    filas = []
    for (p, q, e), g in limpio.groupby(['partido', 'punto', 'equipo']):
        r = {'partido': p, 'punto': q, 'equipo': e,
             'cohesion_media_m': g['dist_companero_m'].mean(), 'cohesion_sd_m': g['dist_companero_m'].std()}
        for var, nombre in [('velocidad_mps', 'sincronia_velocidad'), ('dist_red_m', 'sincronia_profundidad')]:
            ancho = g.pivot_table(index='frame_idx', columns='codigo', values=var)
            with np.errstate(invalid='ignore', divide='ignore'):
                r[nombre] = ancho.iloc[:, 0].corr(ancho.iloc[:, 1]) if ancho.shape[1] == 2 and ancho.dropna().shape[0] >= 10 else np.nan
        filas.append(r)
    return pd.DataFrame(filas)


def tabla_parejas(participaciones, var_jugador, jugadores, puntos, limpio):
    """Una fila por punto completo (D20): variables de cada pareja (e1_, e2_) y equipo 1 − equipo 2 (dif_)."""
    completos = puntos.loc[puntos['equipos_completos'], ['partido', 'punto', 'punto_equipo_1', 'frames', 'fps']]
    completos = completos.assign(duracion_punto_s=(completos['frames'] / completos['fps']).round(2))
    fijas = jugadores[['codigo', *[c for c in VARIABLES_FIJAS if c in jugadores]]]
    jp = (participaciones[['partido', 'punto', 'codigo', 'equipo', 'lado']]
          .merge(completos[['partido', 'punto']], on=['partido', 'punto'])
          .merge(var_jugador, on=LLAVE_JP, how='left').merge(fijas, on='codigo', how='left'))
    jp['en_lado_mano'] = (((jp['zurdo'] == 0) & (jp['lado'] == 'RIGHT')) | ((jp['zurdo'] == 1) & (jp['lado'] == 'LEFT'))).astype(int)
    base = [c for c in VARIABLES_JUGADOR_PAREJA + VARIABLES_FIJAS if c in jp]
    g = jp.groupby(['partido', 'punto', 'equipo'])
    pareja = pd.concat({
        **{f'suma_{c}': g[c].sum(min_count=2) for c in base},
        **{f'absdif_{c}': g[c].agg(lambda s: abs(s.iloc[0] - s.iloc[1]) if s.notna().sum() == 2 else np.nan) for c in base},
        **{f'razon_{c}': g[c].agg(lambda s: s.min() / s.max() if s.notna().sum() == 2 and s.max() > 0 else np.nan)
           for c in VARIABLES_CON_RAZON},
    }, axis=1).reset_index()
    pareja = pareja.merge(_cohesion_y_sincronia(limpio), on=['partido', 'punto', 'equipo'], how='left')
    variables = [c for c in pareja.columns if c not in ['partido', 'punto', 'equipo']]
    e1 = pareja[pareja['equipo'] == 1].set_index(['partido', 'punto'])[variables]
    e2 = pareja[pareja['equipo'] == 2].set_index(['partido', 'punto'])[variables]
    tabla = pd.concat([e1.add_prefix('e1_'), e2.add_prefix('e2_'), (e1 - e2).add_prefix('dif_')], axis=1).reset_index()
    tabla = completos[['partido', 'punto', 'punto_equipo_1', 'duracion_punto_s']].merge(tabla, on=['partido', 'punto'])
    return tabla.sort_values(['partido', 'punto']).reset_index(drop=True)


def filtrar_variables(tabla, excluir=('partido', 'punto', 'punto_equipo_1')):
    """Filtros que no usan el resultado (D33): faltantes > 20 %, valor dominante ≥ 95 % y Pearson |r| ≥ 0,85.

    En cada par muy correlacionado se conserva la variable con menos faltantes; si empatan, se prefiere la
    relativa al rival (dif_), luego la de pareja (e1_/e2_), y después el orden alfabético.
    """
    candidatas = [c for c in tabla.columns if c not in excluir]
    reporte = []
    faltante = tabla[candidatas].isna().mean()
    for c in candidatas:
        if faltante[c] > UMBRAL_FALTANTES:
            reporte.append({'variable': c, 'filtro': 'faltantes', 'detalle': f'{100 * faltante[c]:.0f} % NaN'})
    vivas = [c for c in candidatas if faltante[c] <= UMBRAL_FALTANTES]
    for c in list(vivas):
        s = tabla[c].dropna()
        dominante = s.value_counts(normalize=True).iloc[0] if len(s) else 1.0
        if s.std() == 0 or dominante >= UMBRAL_VALOR_DOMINANTE:
            reporte.append({'variable': c, 'filtro': 'varianza baja', 'detalle': f'valor dominante {100 * dominante:.0f} %'})
            vivas.remove(c)
    prioridad = {c: (faltante[c], 0 if c.startswith('dif_') else 1, c) for c in vivas}
    orden = sorted(vivas, key=lambda c: prioridad[c])
    with np.errstate(invalid='ignore', divide='ignore'):
        corr = tabla[orden].corr(method='pearson').abs()
    conservadas = []
    for c in orden:
        par = next((k for k in conservadas if corr.loc[c, k] >= UMBRAL_CORRELACION), None)
        if par is None:
            conservadas.append(c)
        else:
            reporte.append({'variable': c, 'filtro': 'correlación', 'detalle': f'|r| = {corr.loc[c, par]:.2f} con {par}'})
    return conservadas, pd.DataFrame(reporte)


# ══ 3. Variables para las fases 2 a 4 (1.11) ═══════════════════════════════════
# 19 variables dif_ que se reconstruyen desde cada jugador (D34). dif_suma_mujer sale por |r| = 0,85 con la estatura.
VARIABLES_MODELO = [
    # estilo y posición
    'dif_suma_atacante', 'dif_absdif_atacante', 'dif_absdif_prof_media_m', 'dif_suma_prof_sd_m', 'dif_suma_pct_en_su_lado',
    # movimiento
    'dif_suma_vel_p90', 'dif_razon_vel_p90', 'dif_suma_area_m2', 'dif_razon_area_m2', 'dif_suma_pct_quieto',
    'dif_suma_rafagas_por_min',
    # nivel y atributos fijos (control)
    'dif_suma_nivel_padel', 'dif_absdif_nivel_padel', 'dif_suma_tiempo_jugando_padel', 'dif_suma_practica_otra_raqueta',
    'dif_suma_frecuencia_deporte', 'dif_suma_estatura_cm', 'dif_suma_imc', 'dif_absdif_mujer',
]


def combinar_pareja(j1, j2, variables=VARIABLES_MODELO):
    """De dos jugadores a su pareja: suma, |diferencia| o razón mín/máx según el nombre (D32). Sin el prefijo dif_."""
    pareja = {}
    for v in variables:
        tipo, base = v.removeprefix('dif_').split('_', 1)       # p. ej. 'suma', 'vel_p90'
        a, b = j1[base], j2[base]
        if tipo == 'suma':
            pareja[v] = a + b
        elif tipo == 'absdif':
            pareja[v] = abs(a - b)
        else:  # razón: NaN si falta alguno o si el mayor es 0, como en tabla_parejas
            pareja[v] = np.minimum(a, b) / np.maximum(a, b) if np.maximum(a, b) > 0 else np.nan
    return pareja


def vif(tabla, columnas):
    """VIF de cada columna (diagonal de la inversa de la matriz de correlación), imputando con la mediana."""
    x = tabla[columnas].fillna(tabla[columnas].median())
    return pd.Series(np.diag(np.linalg.inv(x.corr().to_numpy())), index=columnas, name='VIF').sort_values(ascending=False)
