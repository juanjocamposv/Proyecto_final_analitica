"""Limpieza de las fuentes pequeñas: traza de partidos (grano punto) y formulario de jugadores (grano jugador).

Secciones:
1. Traza: controles de consistencia, limpieza, tabla de puntos y formato largo jugador × punto.
2. Formulario: selección de variables, codificación ordinal, imputación (D9) y variables derivadas.
"""

import numpy as np
import pandas as pd

from .identidades import JUGADORES_SIN_FORMULARIO, mapa_alias, nombre_canonico, normalizar_codigo

# ══ 1. Traza ═══════════════════════════════════════════════════════════════════
SLOTS = [(e, j) for e in (1, 2) for j in (1, 2)]   # (equipo, jugador dentro del equipo)
CORRECCIONES_LADO = {'RIGTH': 'RIGHT'}


def validar_traza(traza):
    """Controles de consistencia. Devuelve un DataFrame con el número de filas que fallan cada control."""
    t = traza.sort_values(['ID_PARTIDO', 'ID_PUNTO'])
    m1 = t.groupby('ID_PARTIDO')['PUNTO_EQUIPO_1'].cumsum()
    m2 = t.groupby('ID_PARTIDO')['PUNTO_EQUIPO_2'].cumsum()
    marcador = m1.astype(str) + '-' + m2.astype(str)
    controles = {
        'PUNTO_EQUIPO_1 + PUNTO_EQUIPO_2 != 1': int(((t['PUNTO_EQUIPO_1'] + t['PUNTO_EQUIPO_2']) != 1).sum()),
        'partido-punto duplicado': int(t.duplicated(['ID_PARTIDO', 'ID_PUNTO']).sum()),
        'MARCADOR distinto del acumulado': int((marcador != t['MARCADOR'].astype(str).str.strip()).sum()),
        'ID_PUNTO no consecutivo': int((t.groupby('ID_PARTIDO')['ID_PUNTO'].diff().fillna(1) != 1).sum()),
        'lado LEFT/RIGHT cambia dentro del partido': int(
            t.groupby('ID_PARTIDO')[[f'POSICION_JUGADOR_{j}_EQUIPO_{e}' for e, j in SLOTS]].nunique().gt(1).any(axis=1).sum()),
    }
    return pd.Series(controles, name='filas con problema').to_frame()


def limpiar_traza(traza_cruda, codigos):
    """Traza por punto con textos limpios, lados corregidos y nombres/códigos canónicos por jugador."""
    t = traza_cruda.copy()
    for c in t.select_dtypes(include='object').columns:
        t[c] = t[c].str.strip()
    for e, j in SLOTS:
        t[f'POSICION_JUGADOR_{j}_EQUIPO_{e}'] = t[f'POSICION_JUGADOR_{j}_EQUIPO_{e}'].replace(CORRECCIONES_LADO)
    mapa = mapa_alias()
    cod = codigos.set_index('nombre')['codigo']
    for e, j in SLOTS:
        nombres = t[f'ID_JUGADOR_{j}_EQUIPO_{e}'].map(lambda n: nombre_canonico(n, mapa))
        t[f'NOMBRE_J{j}_E{e}'] = nombres
        t[f'CODIGO_J{j}_E{e}'] = nombres.map(cod)
    return t.sort_values(['ID_PARTIDO', 'ID_PUNTO']).reset_index(drop=True)


def tabla_puntos_traza(traza):
    """Tabla de puntos (una fila por punto) con nombres de columnas del proyecto."""
    base = traza.rename(columns={'ID_PARTIDO': 'partido', 'ID_PUNTO': 'punto', 'PUNTO_EQUIPO_1': 'punto_equipo_1',
                                 'MARCADOR': 'marcador'})
    cols = ['partido', 'punto', 'cancha', 'punto_equipo_1', 'marcador']
    cods = {f'CODIGO_J{j}_E{e}': f'codigo_j{j}_e{e}' for e, j in SLOTS}
    return base[cols + list(cods)].rename(columns=cods)


def traza_larga(traza):
    """Formato largo jugador × punto: equipo, fondo (TOP/BOTTOM), lado (LEFT/RIGHT), compañero y resultado."""
    partes = []
    for e, j in SLOTS:
        otro = 2 if j == 1 else 1
        partes.append(pd.DataFrame({
            'partido': traza['ID_PARTIDO'], 'punto': traza['ID_PUNTO'], 'cancha': traza['cancha'],
            'equipo': e, 'codigo': traza[f'CODIGO_J{j}_E{e}'], 'nombre': traza[f'NOMBRE_J{j}_E{e}'],
            'nombre_traza': traza[f'ID_JUGADOR_{j}_EQUIPO_{e}'],
            'fondo': traza[f'EQUIPO_{e}_POSICION'], 'lado': traza[f'POSICION_JUGADOR_{j}_EQUIPO_{e}'],
            'banda': traza[f'Equipo{e}_Banda{j}'],
            'codigo_companero': traza[f'CODIGO_J{otro}_E{e}'],
            'gano_punto': traza[f'PUNTO_EQUIPO_{e}'].astype(int),
        }))
    return pd.concat(partes, ignore_index=True).sort_values(['partido', 'punto', 'equipo', 'lado']).reset_index(drop=True)


# ══ 2. Formulario ══════════════════════════════════════════════════════════════
# Fuga de datos: nunca predictoras; solo para validar la traza.
COLUMNAS_VALIDACION = ['Puntos_Total', 'Partidos_Jugados', 'Puntos_Por_Partido']

# Escalas ordinales (0 = menor). FRECUENCIA_DEPORTE: '0 veces' y 'No practico deportes' se unen (D12).
ORDINALES = {
    'NIVEL_ACTUAL_PADEL': ('nivel_padel', {'Primera vez': 0, 'Principiante': 1, 'Intermedio': 2, 'Avanzado': 3}),
    'TIEMPO_JUGANDO_PADEL': ('tiempo_jugando_padel', {'Menos de 6 meses': 0, 'Entre 6 meses y 1 año': 1, 'Más de 1 año': 2}),
    'EXPERIENCIA_PADEL': ('experiencia_padel', {'No': 0, 'Sí': 1}),
    'ESTADO_FISICO': ('estado_fisico', {'Malo': 0, 'Regular': 1, 'Bueno': 2, 'Excelente': 3}),
    'FRECUENCIA_DEPORTE': ('frecuencia_deporte', {'No practico deportes': 0, '0 veces': 0, '1-2 veces': 1,
                                                  '3-4 veces': 2, 'Más de 4 veces': 3}),
}
CONTINUAS = {'EDAD': 'edad', 'ESTATURA': 'estatura_cm', 'TALLA': 'peso_kg'}   # TALLA = peso en kg (D8)
CATEGORICAS = {'GENERO': 'genero', 'MANO_DOMINANTE': 'mano_dominante', 'LESIONES_LY': 'lesiones_ly',
               'PRACTICA_OTRO_DEPORTE_RAQUETA': 'otro_deporte_raqueta'}
DECIMALES = {'edad': 0, 'estatura_cm': 0, 'peso_kg': 1}


def limpiar_formulario(formulario):
    """Una fila por jugador con código, nombre canónico y las variables usables ya codificadas."""
    mapa = mapa_alias()
    j = pd.DataFrame({
        'codigo': formulario['CODIGO_ESTUDIANTE'].map(normalizar_codigo),
        'nombre': formulario['NOMBRE'].map(lambda n: nombre_canonico(n, mapa)),
    })
    for origen, destino in CONTINUAS.items():
        j[destino] = pd.to_numeric(formulario[origen], errors='coerce')
    for origen, destino in CATEGORICAS.items():
        j[destino] = formulario[origen].astype('string').str.strip()
    for origen, (destino, escala) in ORDINALES.items():
        valores = formulario[origen].astype('string').str.strip()
        desconocidos = set(valores.dropna()) - set(escala)
        if desconocidos:
            raise ValueError(f'{origen}: categorías sin escala definida {desconocidos}')
        j[destino] = valores.map(escala).astype('Int64')
    j['imputado'] = False
    return j


def imputar_sin_formulario(jugadores, semilla):
    """Agrega los jugadores sin formulario (D9).

    Continuas: media de su género; a las mujeres se les suma ruido N(0, sd de las mujeres) y se
    recorta al rango observado de las mujeres. Ordinales: mediana de su género (si cae entre dos
    categorías, la inferior). Categóricas: moda de su género.
    """
    rng = np.random.default_rng(semilla)
    ordinales = [d for d, _ in ORDINALES.values()]
    nuevas = []
    for nombre, datos in JUGADORES_SIN_FORMULARIO.items():
        if datos['codigo'] in set(jugadores['codigo']):
            continue
        grupo = jugadores[(jugadores['genero'] == datos['genero']) & ~jugadores['imputado']]
        fila = {'codigo': datos['codigo'], 'nombre': nombre, 'genero': datos['genero'], 'imputado': True}
        for c in CONTINUAS.values():
            valor = grupo[c].mean()
            if datos['genero'] == 'Femenino':
                valor = float(np.clip(valor + rng.normal(0, grupo[c].std()), grupo[c].min(), grupo[c].max()))
            fila[c] = round(valor, DECIMALES[c])
        for c in ordinales:
            fila[c] = int(np.floor(grupo[c].astype(float).median()))
        for c in CATEGORICAS.values():
            if c != 'genero':
                fila[c] = grupo[c].mode().sort_values().iloc[0]
        nuevas.append(fila)
    salida = pd.concat([jugadores, pd.DataFrame(nuevas)], ignore_index=True)
    for c in ordinales:
        salida[c] = salida[c].astype('Int64')
    return salida


def agregar_derivadas_jugadores(jugadores):
    """IMC (D8) e indicadores binarios de las categóricas."""
    j = jugadores.copy()
    j['edad'] = j['edad'].round().astype('Int64')
    j['estatura_cm'] = j['estatura_cm'].round().astype('Int64')
    j['imc']= (j['peso_kg'] / (j['estatura_cm'] / 100) ** 2).round(2)
    j['mujer'] = (j['genero'] == 'Femenino').astype(int)
    j['zurdo'] = (j['mano_dominante'] == 'Izquierda').astype(int)
    j['lesion_ultimo_anio'] = (j['lesiones_ly'] == 'Sí').astype(int)
    j['practica_otra_raqueta'] = (j['otro_deporte_raqueta'] != 'No').astype(int)
    orden = ['codigo', 'nombre', 'genero', 'mujer', 'edad', 'estatura_cm', 'peso_kg', 'imc', 'mano_dominante', 'zurdo',
             'nivel_padel', 'tiempo_jugando_padel', 'experiencia_padel', 'otro_deporte_raqueta', 'practica_otra_raqueta',
             'estado_fisico', 'frecuencia_deporte', 'lesiones_ly', 'lesion_ultimo_anio', 'imputado']
    return j[orden].sort_values('nombre').reset_index(drop=True)
