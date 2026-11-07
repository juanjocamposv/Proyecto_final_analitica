"""Video (frame × jugador): calidad, identidad, evidencia de la pelota, atípicos y limpieza.

Secciones:
1. Calidad por punto e identidad de cada track (secciones 1.6 y 1.7). Cada track es
   (partido, punto, player_id); la identidad sale de la etiqueta si es coherente con la traza y,
   si no, de la posición contra la traza (D12).
2. Evidencia para descartar la pelota (sección 1.2.7, D24).
3. Diagnóstico de atípicos de las variables físicas (sección 1.2.8, D23).
4. Limpieza frame × jugador (sección 1.8).
"""

import hashlib

import numpy as np
import pandas as pd
from scipy.signal import find_peaks

from .datos import ANCHO_M, DIST_RED_LINEA_SAQUE_M, LARGO_M, X_CENTRO_M, Y_RED_M
from .identidades import es_generico, mapa_alias, nombre_canonico

LLAVE_PUNTO = ['partido', 'punto']
LLAVE_TRACK = ['partido', 'punto', 'player_id']
COLUMNAS_IDENTIDAD = ['partido', 'punto', 'frame_idx', 'player_id', 'nombre_norm', 'player_position_x',
                      'player_position_y', 'team', 'duration', 'pose_available', 'schema_version']

UMBRAL_Y_COPIA_X = 0.95       # share de filas con x == y para declarar `y` inválida
MIN_FRAMES_TEAM = 5         # frames con `team` para usarlo como fuente del fondo
UMBRAL_CONCORDANCIA = 0.8     # concordancia etiqueta-posición mínima para confiar en las etiquetas del partido


def _a_bool(serie):
    return serie.astype('string').str.lower().eq('true').fillna(False).astype(bool)


def _firma(df):
    s = df.sort_values(['frame_idx', 'player_id'])[['frame_idx', 'player_id', 'player_position_x', 'player_position_y']]
    return hashlib.sha1(pd.util.hash_pandas_object(s, index=False).values.tobytes()).hexdigest()


def diagnosticar_puntos(video):
    """Una fila por punto de video con métricas y banderas de calidad."""
    v = video.assign(
        _generico=video['nombre_norm'].map(es_generico),
        _y_copia_x=np.isclose(video['player_position_x'], video['player_position_y']),
        _fuera=~video['player_position_x'].between(0, ANCHO_M) | ~video['player_position_y'].between(0, LARGO_M),
        _pose=_a_bool(video['pose_available']),
        _team_na=video['team'].isna(),
        _schema=video['schema_version'].notna(),
    )
    d = v.groupby(LLAVE_PUNTO).agg(
        filas=('frame_idx', 'size'), frames=('frame_idx', 'nunique'), tracks=('player_id', 'nunique'),
        fps=('duration', lambda s: round(1 / s.median(), 1)),
        pct_team_faltante=('_team_na', 'mean'), pct_pose=('_pose', 'mean'),
        pct_y_copia_x=('_y_copia_x', 'mean'), pct_fuera_cancha=('_fuera', 'mean'), schema_v2=('_schema', 'any'),
    )
    d.insert(3, 'tracks_genericos', v[v['_generico']].groupby(LLAVE_PUNTO)['player_id'].nunique()
             .reindex(d.index, fill_value=0))
    d['firma'] = v.groupby(LLAVE_PUNTO)[['frame_idx', 'player_id', 'player_position_x', 'player_position_y']].apply(_firma)
    # Duplicados exactos dentro de un partido: se conserva el punto de menor número (D7).
    primero = d.reset_index().groupby(['partido', 'firma'])['punto'].transform('min').values
    d['duplicado_de_punto'] = np.where(primero != d.index.get_level_values('punto'), primero, np.nan)
    d['y_invalida'] = d['pct_y_copia_x'] > UMBRAL_Y_COPIA_X
    d['sin_pose'] = d['pct_pose'] == 0
    for c in ['pct_team_faltante', 'pct_pose', 'pct_y_copia_x', 'pct_fuera_cancha']:
        d[c] = (d[c] * 100).round(1)
    return d.drop(columns='firma').reset_index()


def resumir_tracks(video, diagnostico):
    """Una fila por track con su etiqueta y la evidencia de posición (fondo y lado)."""
    y_inv = diagnostico.set_index(LLAVE_PUNTO)['y_invalida']
    invalida = pd.MultiIndex.from_frame(video[LLAVE_PUNTO]).map(y_inv.to_dict()).fillna(False).astype(bool).to_numpy()
    v = video[LLAVE_TRACK + ['frame_idx', 'nombre_norm', 'player_position_x', 'player_position_y', 'team']].copy()
    v['_top'] = np.where(v['team'].isna(), np.nan, (v['team'] == 'top').astype(float))
    v['_bottom_y'] = np.where(invalida, np.nan, (v['player_position_y'] > Y_RED_M).astype(float))
    v['_right'] = (v['player_position_x'] > X_CENTRO_M).astype(float)
    v.loc[invalida, 'player_position_y'] = np.nan
    t = v.groupby(LLAVE_TRACK).agg(
        etiqueta_video=('nombre_norm', 'first'), etiquetas_distintas=('nombre_norm', 'nunique'),
        filas=('frame_idx', 'size'), x_mediana=('player_position_x', 'median'), y_mediana=('player_position_y', 'median'),
        p_top_team=('_top', 'mean'), n_team=('_top', 'count'), p_bottom_y=('_bottom_y', 'mean'), p_right=('_right', 'mean'),
    ).reset_index()
    if (t['etiquetas_distintas'] > 1).any():
        raise ValueError('Hay tracks con más de una etiqueta de nombre.')
    usar_team = t['n_team'] >= MIN_FRAMES_TEAM
    t['p_bottom'] = np.where(usar_team, 1 - t['p_top_team'], t['p_bottom_y'])
    t['fuente_fondo'] = np.where(usar_team, 'team', np.where(t['p_bottom_y'].notna(), 'y', 'sin dato'))
    t['p_bottom'] = t['p_bottom'].fillna(0.5)
    t['fondo_pos'] = np.where(t['p_bottom'] >= 0.5, 'BOTTOM', 'TOP')
    t['lado_pos'] = np.where(t['p_right'] >= 0.5, 'RIGHT', 'LEFT')
    mapa = mapa_alias()
    t['generico'] = t['etiqueta_video'].map(es_generico)
    t['nombre_etiqueta'] = t['etiqueta_video'].map(lambda n: None if es_generico(n) else nombre_canonico(n, mapa))
    return t.drop(columns='etiquetas_distintas')


def concordancia_por_partido(tracks, participaciones, diagnostico):
    """Por partido: % de filas con etiqueta de la alineación y % cuya posición coincide con la traza."""
    roster = participaciones[['partido', 'punto', 'nombre', 'fondo', 'lado']]
    validos = diagnostico.loc[diagnostico['duplicado_de_punto'].isna(), LLAVE_PUNTO]
    t = tracks.merge(validos, on=LLAVE_PUNTO).merge(
        roster.rename(columns={'nombre': 'nombre_etiqueta', 'fondo': 'fondo_traza', 'lado': 'lado_traza'}),
        on=['partido', 'punto', 'nombre_etiqueta'], how='left')
    con_traza = t.merge(participaciones[LLAVE_PUNTO].drop_duplicates(), on=LLAVE_PUNTO)
    con_traza = con_traza.assign(en_alineacion=con_traza['fondo_traza'].notna(),
                                 concuerda=(con_traza['fondo_pos'] == con_traza['fondo_traza'])
                                 & (con_traza['lado_pos'] == con_traza['lado_traza']))
    def resumen(g):
        en = g[g['en_alineacion']]
        return pd.Series({
            'filas': g['filas'].sum(),
            'pct_filas_en_alineacion': 100 * en['filas'].sum() / g['filas'].sum(),
            'pct_concordancia_posicion': 100 * np.average(en['concuerda'], weights=en['filas']) if len(en) else np.nan,
        })
    r = con_traza.groupby('partido').apply(resumen, include_groups=False)
    confiable = (r['pct_filas_en_alineacion'] >= 50) & (r['pct_concordancia_posicion'] >= 100 * UMBRAL_CONCORDANCIA)
    r['metodo'] = np.where(confiable, 'etiqueta', 'posicion')
    return r.round(1).reset_index()


def _asignar_por_posicion(grupo, roster):
    """Asignación codiciosa track -> casilla (fondo, lado) maximizando filas × p_fondo × p_lado."""
    candidatos = []
    for _, tr in grupo.iterrows():
        for _, slot in roster.iterrows():
            pf = tr['p_bottom'] if slot['fondo'] == 'BOTTOM' else 1 - tr['p_bottom']
            pl = tr['p_right'] if slot['lado'] == 'RIGHT' else 1 - tr['p_right']
            if pf >= 0.5 and pl >= 0.5:
                candidatos.append((tr['filas'] * pf * pl, tr['player_id'], slot['codigo']))
    asignado, usados = {}, set()
    for _, pid, codigo in sorted(candidatos, reverse=True):
        if pid not in asignado and codigo not in usados:
            asignado[pid] = codigo
            usados.add(codigo)
    return asignado


def asignar_identidades(tracks, participaciones, diagnostico, metodo_partido):
    """Tabla de identidades: una fila por track con código asignado, método y motivo si se descarta."""
    diag = diagnostico.set_index(LLAVE_PUNTO)
    metodo = metodo_partido.set_index('partido')['metodo'].to_dict()
    roster_por_punto = {k: g for k, g in participaciones.groupby(LLAVE_PUNTO)}
    salida = []
    for (p, q), grupo in tracks.groupby(LLAVE_PUNTO):
        g = grupo.copy()
        roster = roster_por_punto.get((p, q))
        g['metodo'] = None
        g['codigo'] = None
        dup = diag.loc[(p, q), 'duplicado_de_punto']
        if pd.notna(dup):
            g['motivo'] = f'video duplicado del punto {int(dup)}'
        elif roster is None:
            g['motivo'] = 'punto sin traza'
        elif metodo.get(p, 'posicion') == 'etiqueta':
            cod = roster.set_index('nombre')['codigo']
            g['codigo'] = g['nombre_etiqueta'].map(cod)
            g['metodo'] = 'etiqueta'
            g['motivo'] = np.where(g['codigo'].notna(), None,
                                   np.where(g['generico'], 'track genérico sin identidad', 'nombre fuera de la alineación'))
        else:
            asignado = _asignar_por_posicion(g, roster)
            g['codigo'] = g['player_id'].map(asignado)
            g['metodo'] = 'posicion'
            g['motivo'] = np.where(g['codigo'].notna(), None, 'sin casilla libre compatible')
        salida.append(g)
    ident = pd.concat(salida, ignore_index=True)
    ident['conservar'] = ident['codigo'].notna()
    ident.loc[~ident['conservar'], 'metodo'] = None
    info = participaciones[['partido', 'punto', 'codigo', 'nombre', 'equipo', 'fondo', 'lado']]
    ident = ident.merge(info, on=['partido', 'punto', 'codigo'], how='left')
    ident['concuerda_posicion'] = np.where(ident['conservar'], (ident['fondo_pos'] == ident['fondo'])
                                           & (ident['lado_pos'] == ident['lado']), np.nan)
    cols = ['partido', 'punto', 'player_id', 'etiqueta_video', 'filas', 'x_mediana', 'y_mediana', 'fuente_fondo',
            'p_bottom', 'p_right', 'fondo_pos', 'lado_pos', 'conservar', 'metodo', 'motivo', 'codigo', 'nombre',
            'equipo', 'fondo', 'lado', 'concuerda_posicion']
    out = ident[cols].sort_values(LLAVE_TRACK).reset_index(drop=True)
    for c in ['x_mediana', 'y_mediana', 'p_bottom', 'p_right']:
        out[c] = out[c].round(3)
    return out


def enlazar_participaciones(participaciones, identidades):
    """Agrega a cada jugador × punto de la traza su track de video (player_id) y filas, si existe."""
    ok = identidades.loc[identidades['conservar'], ['partido', 'punto', 'codigo', 'player_id', 'filas', 'metodo']]
    p = participaciones.merge(ok.rename(columns={'filas': 'filas_video', 'metodo': 'metodo_identidad'}),
                              on=['partido', 'punto', 'codigo'], how='left')
    p['player_id'] = p['player_id'].astype('Int64')
    p['filas_video'] = p['filas_video'].fillna(0).astype(int)
    return p


def tabla_puntos(puntos_traza, diagnostico, identidades):
    """Unión externa traza–video por punto con estado del video e identificados por equipo."""
    ok = identidades[identidades['conservar']]
    por_equipo = ok.pivot_table(index=LLAVE_PUNTO, columns='equipo', values='codigo', aggfunc='nunique', fill_value=0)
    por_equipo = por_equipo.reindex(columns=[1, 2], fill_value=0)
    por_equipo.columns = ['identificados_e1', 'identificados_e2']
    p = puntos_traza.assign(en_traza=True).merge(diagnostico.assign(tiene_video=True), on=LLAVE_PUNTO, how='outer')
    p = p.merge(por_equipo.reset_index(), on=LLAVE_PUNTO, how='left')
    for c in ['en_traza', 'tiene_video']:
        p[c] = p[c].astype('boolean').fillna(False).astype(bool)
    p[['identificados_e1', 'identificados_e2']] = p[['identificados_e1', 'identificados_e2']].fillna(0).astype(int)
    p['estado_video'] = np.select(
        [~p['tiene_video'], ~p['en_traza'], p['duplicado_de_punto'].notna()],
        ['sin video', 'video sin traza', 'duplicado'], default='disponible')
    p['equipos_completos'] = (p['identificados_e1'] == 2) & (p['identificados_e2'] == 2)
    p['un_solo_fondo'] = (p['estado_video'] == 'disponible') & ((p['identificados_e1'] == 0) | (p['identificados_e2'] == 0))
    return p.sort_values(LLAVE_PUNTO).reset_index(drop=True)


# ══ 2. Evidencia para descartar la pelota (sección 1.2.7, D24) ═════════════════
COLUMNAS_PELOTA = ['frame_idx', 'player_id', 'ball_position_x', 'ball_position_y', 'player_hits_ball',
                   'distance_player_to_ball_m', 'racket_speed', 'body_scale']


def clasificar_frames_pelota(video):
    """Clasifica cada frame de la pelota: quieta (relleno), tramo recto (interpolada) o posible detección.

    Una posición de relleno no cambia entre frames; una interpolación lineal tiene segunda diferencia 0.
    """
    b = (video[LLAVE_PUNTO + ['frame_idx', 'ball_position_x', 'ball_position_y']]
         .drop_duplicates(LLAVE_PUNTO + ['frame_idx']).sort_values(LLAVE_PUNTO + ['frame_idx']))
    claves = [b['partido'], b['punto']]
    dx = b.groupby(LLAVE_PUNTO)['ball_position_x'].diff()
    dy = b.groupby(LLAVE_PUNTO)['ball_position_y'].diff()
    d2x, d2y = dx.groupby(claves).diff(), dy.groupby(claves).diff()
    clase = np.select(
        [b['ball_position_x'].isna(), d2x.isna(), (dx.abs() + dy.abs()) < 1e-9,
         (d2x.abs() < 1e-6) & (d2y.abs() < 1e-6)],
        ['sin dato', 'inicio del punto', 'quieta (relleno)', 'tramo recto (interpolada)'], 'posible detección')
    conteo = pd.Series(clase).value_counts()
    validos = conteo.drop(['sin dato', 'inicio del punto'], errors='ignore')
    return validos.rename('frames').to_frame().assign(pct=lambda d: (100 * d['frames'] / d['frames'].sum()).round(1))


def golpes_frente_a_picos(video, ventana_frames=5):
    """¿Coincide `player_hits_ball` con picos de velocidad de raqueta y con la cercanía a la pelota?"""
    v = video.sort_values(LLAVE_TRACK + ['frame_idx']).copy()
    golpe = _a_bool(v['player_hits_ball'])
    rank = (v['racket_speed'] / v['body_scale']).groupby([v[c] for c in LLAVE_TRACK]).rank(pct=True)
    pico = (rank > 0.95).astype(float)
    cerca = pico.groupby([v[c] for c in LLAVE_TRACK]).transform(
        lambda s: s.rolling(2 * ventana_frames + 1, center=True, min_periods=1).max())
    return pd.DataFrame([
        {'medida': 'filas con player_hits_ball', 'valor': int(golpe.sum())},
        {'medida': 'partidos con algún golpe', 'valor': int(v.loc[golpe, 'partido'].nunique())},
        {'medida': 'distancia mediana jugador–pelota en golpes (m)', 'valor': round(v.loc[golpe, 'distance_player_to_ball_m'].median(), 2)},
        {'medida': 'distancia mediana jugador–pelota en el resto (m)', 'valor': round(v.loc[~golpe, 'distance_player_to_ball_m'].median(), 2)},
        {'medida': f'% de golpes con pico de raqueta (top 5 %) a ±{ventana_frames} frames', 'valor': round(100 * cerca[golpe].mean(), 1)},
        {'medida': f'% de frames cualesquiera con pico a ±{ventana_frames} frames (azar)', 'valor': round(100 * cerca.mean(), 1)},
    ])


# ══ 3. Atípicos de las variables físicas (sección 1.2.8, D23) ══════════════════
VENTANA_SUAVIZADO_S = 0.5            # media móvil centrada de la posición (decisión 1.8)
LIMITE_VELOCIDAD_MPS = 8.0           # por encima de lo que alcanza un jugador de pádel
LIMITE_ACELERACION_MPS2 = 10.0
LIMITE_ANGULAR_DPS = 3000.0          # velocidad angular del codo o de la raqueta
PERCENTIL_PIXELES = 0.995            # velocidades en píxeles (normalizadas): percentil por jugador
MAX_HUECO_FRAMES = 5                 # huecos más largos quedan NaN
VARIABLES_PIXELES = {'racket_speed': 'vel_raqueta_norm', 'wrist_speed': 'vel_muneca_norm', 'elbow_speed': 'vel_codo_norm'}
COLUMNAS_ATIPICOS = ['frame_idx', 'player_id', 'duration', 'player_position_x', 'player_position_y', 'player_speed_mps',
                     'player_acceleration_mps2', 'elbow_angular_velocity', 'racket_angular_velocity', 'body_scale',
                     *VARIABLES_PIXELES]


def _ventana(fps):
    return max(3, int(round(VENTANA_SUAVIZADO_S * fps)))


def _suavizar(s, w):
    return s.rolling(w, center=True, min_periods=max(1, w // 2)).mean()


def cinematica(x, y, duracion):
    """Posición suavizada, velocidad (m/s) y aceleración (m/s²) de un track ordenado por frame."""
    w = _ventana(1 / float(np.nanmedian(duracion)))
    xs, ys = _suavizar(x, w), _suavizar(y, w)
    velocidad = _suavizar(np.hypot(xs.diff(), ys.diff()) / duracion, w)
    aceleracion = _suavizar(velocidad.diff() / duracion, w)
    return xs, ys, velocidad, aceleracion


def diagnosticar_atipicos(video):
    """Rangos de las variables físicas crudas y suavizadas frente a los límites (D23)."""
    v = video.sort_values(LLAVE_TRACK + ['frame_idx'])
    vel, acc = [], []
    for _, g in v.groupby(LLAVE_TRACK):
        _, _, ve, ac = cinematica(g['player_position_x'], g['player_position_y'], g['duration'])
        vel.append(ve)
        acc.append(ac)
    suav_v, suav_a = pd.concat(vel), pd.concat(acc)

    def fila(nombre, serie, limite, criterio):
        a = serie.astype(float).abs()
        return {'variable': nombre, 'criterio': criterio, 'limite': round(limite, 1), 'pct_faltante': round(100 * a.isna().mean(), 1),
                'p50': round(a.median(), 2), 'p99': round(a.quantile(0.99), 2), 'max': round(a.max(), 1),
                'pct_sobre_limite': round(100 * (a > limite).sum() / max(a.notna().sum(), 1), 2)}

    filas = [
        fila('player_speed_mps (cruda)', v['player_speed_mps'], LIMITE_VELOCIDAD_MPS, 'físico'),
        fila('player_acceleration_mps2 (cruda)', v['player_acceleration_mps2'], LIMITE_ACELERACION_MPS2, 'físico'),
        fila(f'velocidad suavizada {VENTANA_SUAVIZADO_S} s', suav_v, LIMITE_VELOCIDAD_MPS, 'físico'),
        fila(f'aceleración suavizada {VENTANA_SUAVIZADO_S} s', suav_a, LIMITE_ACELERACION_MPS2, 'físico'),
        fila('elbow_angular_velocity (°/s)', v['elbow_angular_velocity'], LIMITE_ANGULAR_DPS, 'físico'),
        fila('racket_angular_velocity (°/s)', v['racket_angular_velocity'], LIMITE_ANGULAR_DPS, 'físico'),
    ]
    for col, nombre in VARIABLES_PIXELES.items():
        norm = v[col] / v['body_scale']
        filas.append(fila(f'{nombre} ({col} / body_scale)', norm, norm.quantile(PERCENTIL_PIXELES), 'percentil 99,5 (global; por jugador en 1.8)'))
    return pd.DataFrame(filas)


# ══ 4. Limpieza frame × jugador (sección 1.8) ══════════════════════════════════
MARGEN_CANCHA_M = 0.5                # hasta 0,5 m afuera se recorta al borde; más lejos, NaN
LIMITE_RED_M = 3.0                   # secciones descriptivas: red < 3 m ≤ transición < 6,95 m ≤ fondo
ZONA_ATACANTE_M = DIST_RED_LINEA_SAQUE_M   # estilo (D21): atacante = más tiempo delante de la línea de saque
GOLPE_PERCENTIL = 0.90               # pico de vel_raqueta_norm sobre el p90 del jugador en el punto
GOLPE_SEPARACION_S = 0.4
COLUMNAS_LIMPIEZA = ['frame_idx', 'player_id', 'duration', 'player_position_x', 'player_position_y', 'pose_available',
                     'pose_confidence', 'elbow_angle', 'shoulder_angle', 'racket_orientation_deg', 'racket_direction',
                     'wrist_above_head', 'racket_above_shoulder', 'wrist_height_norm', 'racket_height_norm', 'body_scale',
                     'elbow_angular_velocity', 'racket_angular_velocity', *VARIABLES_PIXELES]
POSE_NUMERICAS = {'pose_confidence': 'confianza_pose', 'elbow_angle': 'angulo_codo', 'shoulder_angle': 'angulo_hombro',
                  'racket_orientation_deg': 'orientacion_raqueta', 'wrist_height_norm': 'altura_muneca_norm',
                  'racket_height_norm': 'altura_raqueta_norm'}


def interpolar_huecos(s, max_hueco=MAX_HUECO_FRAMES):
    """Interpola solo los huecos internos de hasta `max_hueco` frames; los más largos quedan NaN."""
    nan = s.isna()
    if not nan.any():
        return s
    bloque = (nan != nan.shift()).cumsum()
    largo = nan.groupby(bloque).transform('sum')
    return s.where(~(nan & (largo <= max_hueco)), s.interpolate(limit_area='inside'))


def _posicion_en_cancha(s, maximo):
    """Recorta al borde lo que está a ≤ MARGEN_CANCHA_M afuera; lo que está más lejos pasa a NaN."""
    fuera_lejos = (s < -MARGEN_CANCHA_M) | (s > maximo + MARGEN_CANCHA_M)
    return s.clip(0, maximo).mask(fuera_lejos)


def _limpiar_track(g, y_invalida):
    g = g.sort_values('frame_idx').reset_index(drop=True)
    dur = g['duration']
    fps = 1 / float(dur.median())
    x = _posicion_en_cancha(g['player_position_x'], ANCHO_M)
    if y_invalida:
        y = pd.Series(np.nan, index=g.index)
        fuera = x.isna() & g['player_position_x'].notna()
    else:
        y = _posicion_en_cancha(g['player_position_y'], LARGO_M)
        fuera = (x.isna() & g['player_position_x'].notna()) | (y.isna() & g['player_position_y'].notna())
    # Pasada 1: frames con velocidad suavizada imposible (saltos del tracking) pierden la posición.
    # Con y inválida la velocidad 2D no existe: queda NaN y no se marcan saltos.
    _, _, vel, _ = cinematica(x, y, dur)
    salto = (vel > LIMITE_VELOCIDAD_MPS).to_numpy()
    x, y = interpolar_huecos(x.mask(salto)), interpolar_huecos(y.mask(salto))
    # Pasada 2: cinemática definitiva; velocidad o aceleración fuera de límite -> NaN y reconstrucción.
    xs, ys, vel, acc = cinematica(x, y, dur)
    atipica = (vel > LIMITE_VELOCIDAD_MPS) | (acc.abs() > LIMITE_ACELERACION_MPS2)
    vel, acc = interpolar_huecos(vel.mask(atipica)), interpolar_huecos(acc.mask(atipica))
    out = pd.DataFrame({
        'frame_idx': g['frame_idx'], 'tiempo_s': (g['frame_idx'] - g['frame_idx'].min()) * float(dur.median()),
        'fps': round(fps, 2), 'x_m': xs, 'y_m': ys, 'velocidad_mps': vel, 'aceleracion_mps2': acc,
        'desplazamiento_m': vel * dur,
        'posicion_corregida': fuera.to_numpy() | salto, 'cinematica_corregida': atipica.to_numpy(),
        'pose_disponible': _a_bool(g['pose_available']).to_numpy(),
    })
    for col, nombre in POSE_NUMERICAS.items():
        out[nombre] = interpolar_huecos(g[col].astype(float))
    out['direccion_raqueta'] = g['racket_direction'].astype('string')
    out['raqueta_sobre_hombro'] = _bool_con_nulos(g['racket_above_shoulder'])
    out['muneca_sobre_cabeza'] = _bool_con_nulos(g['wrist_above_head'])
    for col, nombre in [('elbow_angular_velocity', 'vel_ang_codo_dps'), ('racket_angular_velocity', 'vel_ang_raqueta_dps')]:
        s = g[col].astype(float)
        out[nombre] = interpolar_huecos(s.mask(s.abs() > LIMITE_ANGULAR_DPS))
    for col, nombre in VARIABLES_PIXELES.items():
        out[nombre] = (g[col] / g['body_scale']).to_numpy()
    return out


def _bool_con_nulos(serie):
    """Texto/bool `True`/`False` con faltantes -> boolean de pandas (NA donde falta)."""
    t = serie.astype('string').str.lower()
    return t.map({'true': True, 'false': False}).astype('boolean')


def limpiar_video(video, identidades, diagnostico):
    """Tabla limpia frame × jugador de los tracks conservados (sección 1.8).

    Por track: posición en cancha (margen 0,5 m), saltos del tracking, suavizado de 0,5 s, velocidad y
    aceleración con límites físicos, pose interpolada en huecos ≤5 frames y velocidades en píxeles
    normalizadas por body_scale. Luego: percentil 99,5 por jugador en píxeles, variables combinadas
    (D25), rol, profundidad, distancia al compañero y golpes.
    """
    ids = identidades[identidades['conservar']].set_index(LLAVE_TRACK)
    y_inv = diagnostico.set_index(LLAVE_PUNTO)['y_invalida']
    v = video.merge(ids[[]].reset_index(), on=LLAVE_TRACK)
    partes = []
    for (p, q, pid), g in v.groupby(LLAVE_TRACK):
        t = _limpiar_track(g, bool(y_inv.loc[(p, q)]))
        info = ids.loc[(p, q, pid)]
        t.insert(0, 'partido', p)
        t.insert(1, 'punto', q)
        t.insert(3, 'player_id', pid)
        t['codigo'], t['equipo'], t['fondo'], t['lado'] = info['codigo'], int(info['equipo']), info['fondo'], info['lado']
        partes.append(t)
    lim = pd.concat(partes, ignore_index=True)

    # Velocidades en píxeles: percentil 99,5 de cada jugador -> NaN y reconstrucción por track.
    for nombre in VARIABLES_PIXELES.values():
        umbral = lim.groupby('codigo')[nombre].transform(lambda s: s.quantile(PERCENTIL_PIXELES))
        lim[nombre] = lim[nombre].mask(lim[nombre] > umbral)
        lim[nombre] = lim.groupby(LLAVE_TRACK)[nombre].transform(interpolar_huecos)

    # Variables combinadas (D25), rol (lado de la traza) y profundidad (D21).
    lim['indice_vel_golpe'] = lim[list(VARIABLES_PIXELES.values())].mean(axis=1)
    rh, mc = lim['raqueta_sobre_hombro'], lim['muneca_sobre_cabeza']
    lim['juego_aereo'] = (rh.fillna(False) | mc.fillna(False)).astype('boolean').mask(rh.isna() & mc.isna())
    lim['team'] = lim['fondo'].str.lower()
    lim['rol'] = np.where(lim['lado'] == 'RIGHT', 'drive', 'reves')
    lim['lado_real'] = pd.Series(np.where(lim['x_m'] > X_CENTRO_M, 'RIGHT', 'LEFT'), dtype='string').mask(lim['x_m'].isna())
    lim['en_su_lado'] = (lim['lado_real'] == lim['lado']).astype('boolean').mask(lim['x_m'].isna())
    lim['dist_red_m'] = (lim['y_m'] - Y_RED_M).abs()
    lim['seccion'] = pd.cut(lim['dist_red_m'], [-np.inf, LIMITE_RED_M, DIST_RED_LINEA_SAQUE_M, np.inf],
                            right=False, labels=['red', 'transicion', 'fondo']).astype('string')

    # Distancia al compañero, recalculada con las posiciones suavizadas.
    comp = lim[['partido', 'punto', 'frame_idx', 'equipo', 'codigo', 'x_m', 'y_m']]
    par = comp.merge(comp, on=['partido', 'punto', 'frame_idx', 'equipo'], suffixes=('', '_c'))
    par = par[par['codigo'] != par['codigo_c']]
    par = par.assign(dist_companero_m=np.hypot(par['x_m'] - par['x_m_c'], par['y_m'] - par['y_m_c']))
    lim = lim.merge(par[['partido', 'punto', 'frame_idx', 'codigo', 'dist_companero_m']],
                    on=['partido', 'punto', 'frame_idx', 'codigo'], how='left')

    # Golpes: picos de vel_raqueta_norm sobre el p90 del jugador en el punto, separados ≥ 0,4 s.
    lim = lim.sort_values(LLAVE_TRACK + ['frame_idx']).reset_index(drop=True)
    golpe = np.zeros(len(lim), dtype=bool)
    for _, idx in lim.groupby(LLAVE_TRACK).indices.items():
        s = lim['vel_raqueta_norm'].iloc[idx]
        if s.notna().sum() < 3:
            continue
        sep = max(1, int(round(GOLPE_SEPARACION_S * float(lim['fps'].iloc[idx[0]]))))
        picos, _ = find_peaks(s.fillna(0).to_numpy(), height=s.quantile(GOLPE_PERCENTIL), distance=sep)
        golpe[idx[picos]] = True
    lim['golpe'] = golpe

    orden = ['partido', 'punto', 'frame_idx', 'player_id', 'codigo', 'equipo', 'fondo', 'lado', 'rol', 'team', 'tiempo_s',
             'fps', 'x_m', 'y_m', 'posicion_corregida', 'dist_red_m', 'seccion', 'lado_real', 'en_su_lado',
             'dist_companero_m', 'velocidad_mps', 'aceleracion_mps2', 'desplazamiento_m', 'cinematica_corregida',
             'pose_disponible', *POSE_NUMERICAS.values(), 'direccion_raqueta', 'raqueta_sobre_hombro',
             'muneca_sobre_cabeza', 'juego_aereo', 'vel_ang_codo_dps', 'vel_ang_raqueta_dps',
             *VARIABLES_PIXELES.values(), 'indice_vel_golpe', 'golpe']
    return lim[orden]


def calidad_limpieza(limpio):
    """Resumen por jugador × punto: frames, cobertura de pose, % corregido, % sin y, y golpes detectados."""
    r = limpio.groupby(['partido', 'punto', 'codigo']).agg(
        frames_video=('frame_idx', 'size'), pose_coverage=('pose_disponible', 'mean'),
        pct_posicion_corregida=('posicion_corregida', 'mean'), pct_cinematica_corregida=('cinematica_corregida', 'mean'),
        pct_y_faltante=('y_m', lambda s: s.isna().mean()), golpes=('golpe', 'sum'),
    ).reset_index()
    for c in ['pose_coverage', 'pct_posicion_corregida', 'pct_cinematica_corregida', 'pct_y_faltante']:
        r[c] = (100 * r[c]).round(1)
    return r
