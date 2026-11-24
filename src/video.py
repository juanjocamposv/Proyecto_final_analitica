"""Video (frame × jugador): identidad de cada track, faltantes, evidencia de la pelota y limpieza.

Secciones:
1. Calidad por punto e identidad de cada track (sección 1.2). Cada track es (partido, punto, player_id);
   la identidad sale de la etiqueta si es coherente con la traza y, si no, de la posición contra la traza (D17).
2. Faltantes por familia de columnas y evidencia para descartar la pelota (sección 1.2, D14).
3. Limpieza frame × jugador: solo posición y velocidad (sección 1.2, D20, D15).
"""

import hashlib

import numpy as np
import pandas as pd

from .datos import ANCHO_M, LARGO_M, X_CENTRO_M, Y_RED_M, iterar_video
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
    # Duplicados exactos dentro de un partido: se conserva el punto de menor número (D16).
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


# ══ 2. Faltantes por familia y evidencia de la pelota (sección 1.2, D14) ════════
PREFIJOS_POSE = {'head', 'shoulder', 'elbow', 'wrist', 'racket', 'upper', 'forearm', 'arm', 'body', 'pose'}
POSICION = {'player_position_x', 'player_position_y', 'distance_player_to_net_m', 'distance_player_to_teammate_m',
            'zone', 'prev_x', 'prev_y', 'player_displacement', 'player_speed_mps', 'prev_speed', 'player_acceleration_mps2'}
TRATAMIENTO = {
    'identificación y tiempo': 'Usar; `team` sale de la traza',
    'posición y movimiento': 'Limpiar y recalcular (D20)',
    'pose y raqueta': 'Eliminar (D15)',
    'pelota': 'Eliminar (D14)',
    'controles de plausibilidad': 'Eliminar',
}


def familia(columna):
    """Familia de una columna del video (diccionario, parte A1)."""
    if 'ball' in columna or columna == 'time_since_last_hit':
        return 'pelota'
    if columna.endswith('_implausible') or columna == 'schema_version':
        return 'controles de plausibilidad'
    if columna.split('_')[0] in PREFIJOS_POSE:
        return 'pose y raqueta'
    if columna in POSICION:
        return 'posición y movimiento'
    return 'identificación y tiempo'


def faltantes_por_familia():
    """% de vacíos por familia de columnas en el video completo (se recorre partido por partido)."""
    vacios, filas = 0, 0
    for bloque in iterar_video():
        vacios = vacios + bloque.isna().sum()
        filas += len(bloque)
    pct = 100 * vacios / filas
    t = pct.groupby(pct.index.map(familia)).agg(columnas='size', pct_vacio_min='min', pct_vacio_max='max').round(1)
    return t.reindex(list(TRATAMIENTO)).assign(tratamiento=list(TRATAMIENTO.values()))


COLUMNAS_PELOTA = ['frame_idx', 'player_id', 'ball_position_x', 'ball_position_y']


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


# ══ 3. Limpieza frame × jugador: posición y velocidad (sección 1.2) ═══════════
VENTANA_SUAVIZADO_S = 0.5            # media móvil centrada de la posición (D20)
LIMITE_VELOCIDAD_MPS = 8.0           # por encima de lo que alcanza un jugador de pádel (D20)
LIMITE_ACELERACION_MPS2 = 10.0
MAX_HUECO_FRAMES = 5                 # huecos más largos quedan NaN
MARGEN_CANCHA_M = 0.5                # hasta 0,5 m afuera se recorta al borde; más lejos, NaN (D20)
COLUMNAS_LIMPIEZA = ['frame_idx', 'player_id', 'duration', 'player_position_x', 'player_position_y']


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
    x = _posicion_en_cancha(g['player_position_x'], ANCHO_M)
    if y_invalida:                                     # y = x en el punto: la y queda vacía (D3)
        y = pd.Series(np.nan, index=g.index)
        fuera = x.isna() & g['player_position_x'].notna()
    else:
        y = _posicion_en_cancha(g['player_position_y'], LARGO_M)
        fuera = (x.isna() & g['player_position_x'].notna()) | (y.isna() & g['player_position_y'].notna())
    # Pasada 1: frames con velocidad suavizada imposible (saltos del tracking) pierden la posición.
    _, _, vel, _ = cinematica(x, y, dur)
    salto = (vel > LIMITE_VELOCIDAD_MPS).to_numpy()
    x, y = interpolar_huecos(x.mask(salto)), interpolar_huecos(y.mask(salto))
    # Pasada 2: cinemática definitiva; velocidad o aceleración fuera de límite -> NaN y reconstrucción.
    xs, ys, vel, acc = cinematica(x, y, dur)
    atipica = (vel > LIMITE_VELOCIDAD_MPS) | (acc.abs() > LIMITE_ACELERACION_MPS2)
    vel, acc = interpolar_huecos(vel.mask(atipica)), interpolar_huecos(acc.mask(atipica))
    return pd.DataFrame({
        'frame_idx': g['frame_idx'], 'fps': round(1 / float(dur.median()), 2), 'x_m': xs, 'y_m': ys,
        'dist_red_m': (ys - Y_RED_M).abs(), 'velocidad_mps': vel, 'aceleracion_mps2': acc,
        'posicion_corregida': fuera.to_numpy() | salto, 'cinematica_corregida': atipica.to_numpy(),
    })


def limpiar_video(video, identidades, diagnostico):
    """Tabla limpia frame × jugador de los tracks conservados: posición en cancha, saltos del tracking,
    suavizado de 0,5 s y velocidad y aceleración con límites físicos (D20)."""
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
        t.insert(4, 'codigo', info['codigo'])
        t.insert(5, 'equipo', int(info['equipo']))
        t.insert(6, 'fondo', info['fondo'])
        t.insert(7, 'lado', info['lado'])
        partes.append(t)
    return pd.concat(partes, ignore_index=True)
