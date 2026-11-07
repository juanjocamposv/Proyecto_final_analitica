"""CSV de video (frame × jugador): diagnóstico de calidad por punto y asignación de identidades.

Cada track es (partido, punto, player_id). La identidad se toma de la etiqueta de nombre cuando
esta es coherente con la traza; si no, se asigna por posición (fondo y lado) contra la traza (D12).
"""

import hashlib

import numpy as np
import pandas as pd

from .datos import ANCHO_M, LARGO_M, X_CENTRO_M, Y_RED_M
from .identidades import es_generico, mapa_alias, nombre_canonico

LLAVE_PUNTO = ['partido', 'punto']
LLAVE_TRACK = ['partido', 'punto', 'player_id']
COLUMNAS_IDENTIDAD = ['partido', 'punto', 'frame_idx', 'player_id', 'nombre_norm', 'player_position_x',
                      'player_position_y', 'team', 'duration', 'pose_available', 'schema_version']

UMBRAL_Y_COPIA_X = 0.95       # share de filas con x == y para declarar `y` inválida
MIN_FRAMES_TEAM = 5         # frames con `team` para usarlo como fuente del fondo
UMBRAL_CONCORDANCIA = 0.8     # concordancia etiqueta-posición mínima para confiar en las etiquetas del partido
FPS_ESTANDAR = (24, 30, 60)


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
    d['y_mediana_top'] = v[v['team'] == 'top'].groupby(LLAVE_PUNTO)['player_position_y'].median()
    d['firma'] = v.groupby(LLAVE_PUNTO)[['frame_idx', 'player_id', 'player_position_x', 'player_position_y']].apply(_firma)
    # Duplicados exactos dentro de un partido: se conserva el punto de menor número (D7).
    primero = d.reset_index().groupby(['partido', 'firma'])['punto'].transform('min').values
    d['duplicado_de_punto'] = np.where(primero != d.index.get_level_values('punto'), primero, np.nan)
    d['y_invalida'] = d['pct_y_copia_x'] > UMBRAL_Y_COPIA_X
    d['sin_pose'] = d['pct_pose'] == 0
    d['y_top_distorsionada'] = d['y_mediana_top'] > Y_RED_M
    d['fps_no_estandar'] = ~d['fps'].round().isin(FPS_ESTANDAR) & ~d['fps'].sub(59.9).abs().lt(0.2)
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
