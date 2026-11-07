"""Base del pipeline: rutas, constantes de la cancha, lectura de las fuentes y utilidades de reporte.

Secciones:
1. Rutas y parámetros globales.
2. Constantes de la cancha (sistema de coordenadas del CSV de video).
3. Lectura de las tres fuentes crudas (sin transformar su contenido).
4. Utilidades: resumen antes/después de cada paso y escritura de tablas derivadas.
"""

from pathlib import Path

import pandas as pd

# ── 1. Rutas y parámetros ──────────────────────────────────────────────────────
RAIZ = Path(__file__).resolve().parents[1]

DATA_RAW = RAIZ / 'data' / 'raw'
DATA_INTERIM = RAIZ / 'data' / 'interim'
DATA_PROCESSED = RAIZ / 'data' / 'processed'

ARCHIVO_VIDEO = DATA_RAW / 'Base_Videos_Final_202602.csv'
ARCHIVO_TRAZA = DATA_RAW / 'Traza_Partidos_202602.xlsx'
ARCHIVO_FORMULARIO = DATA_RAW / 'Datos_Jugadores_Padel_202602.xlsx'

# Hojas útiles de los Excel. Las demás (TDs, Resultados, Hoja 1) son agregados y no se usan.
HOJAS_TRAZA = {1: 'CANCHA_1', 2: 'CANCHA_2'}
HOJA_PROGRAMACION = 'Original'
HOJA_FORMULARIO = 'Respuestas de formulario 1'

# Semilla única para todo proceso aleatorio (imputación, validación, simulación).
SEMILLA = 2026

# ── 2. Constantes de la cancha ─────────────────────────────────────────────────
# x en [0, ANCHO_M] a lo ancho, y en [0, LARGO_M] a lo largo; (0, 0) es la esquina del fondo lejano izquierdo.
# Convención verificada contra la traza (D13): BOTTOM está en y > Y_RED_M y, para ambos equipos,
# RIGHT está en x > X_CENTRO_M (vista de la cámara).
ANCHO_M = 10.0
LARGO_M = 20.0
Y_RED_M = LARGO_M / 2
X_CENTRO_M = ANCHO_M / 2
DIST_RED_LINEA_SAQUE_M = 6.95   # de la red a la línea de saque (límite de los cuadros de servicio)


# ── 3. Lectura de las fuentes ──────────────────────────────────────────────────
def leer_traza(ruta=None):
    """Une las hojas CANCHA_1 y CANCHA_2 de la traza y agrega la columna `cancha`."""
    ruta = ruta or ARCHIVO_TRAZA
    hojas = [pd.read_excel(ruta, sheet_name=hoja).assign(cancha=cancha) for cancha, hoja in HOJAS_TRAZA.items()]
    traza = pd.concat(hojas, ignore_index=True)
    traza.columns = [str(c).strip() for c in traza.columns]
    return traza.dropna(how='all').reset_index(drop=True)


def leer_programacion(ruta=None):
    """Hoja `Original`: los partidos programados con parejas y bandas (incluye partidos sin traza)."""
    return pd.read_excel(ruta or ARCHIVO_TRAZA, sheet_name=HOJA_PROGRAMACION)


def leer_formulario(ruta=None):
    """Respuestas del formulario de jugadores (una fila por jugador)."""
    formulario = pd.read_excel(ruta or ARCHIVO_FORMULARIO, sheet_name=HOJA_FORMULARIO)
    formulario.columns = [str(c).strip() for c in formulario.columns]
    return formulario.dropna(how='all').reset_index(drop=True)


def columnas_video(ruta=None):
    """Nombres de las columnas del CSV de video sin cargar sus filas."""
    return list(pd.read_csv(ruta or ARCHIVO_VIDEO, nrows=0).columns)


def leer_video(columnas=None, partidos=None, ruta=None, chunksize=200_000):
    """Lee el CSV consolidado de video (frame × jugador).

    columnas: subconjunto de columnas (recomendado: el archivo completo ocupa ~650 MB).
    partidos: si se indica, solo conserva esos partidos (lectura por bloques).
    """
    ruta = ruta or ARCHIVO_VIDEO
    if columnas is not None:
        columnas = list(dict.fromkeys(list(columnas) + ['partido', 'punto']))
    if partidos is None:
        return pd.read_csv(ruta, usecols=columnas, low_memory=False)
    partidos = set(partidos)
    bloques = [b[b['partido'].isin(partidos)]
               for b in pd.read_csv(ruta, usecols=columnas, low_memory=False, chunksize=chunksize)]
    return pd.concat(bloques, ignore_index=True)


def iterar_video(chunksize=200_000, columnas=None, ruta=None):
    """Recorre el CSV de video por bloques, para perfiles del archivo completo sin cargarlo entero."""
    return pd.read_csv(ruta or ARCHIVO_VIDEO, usecols=columnas, chunksize=chunksize, low_memory=False)


# ── 4. Utilidades de reporte y escritura ───────────────────────────────────────
def resumen_tabla(df, nombre):
    """Una fila con filas, columnas y faltantes: se muestra antes y después de cada paso."""
    celdas = df.shape[0] * df.shape[1]
    faltantes = int(df.isna().sum().sum())
    return pd.DataFrame([{
        'tabla': nombre, 'filas': len(df), 'columnas': df.shape[1], 'celdas_faltantes': faltantes,
        'pct_faltante': round(100 * faltantes / celdas, 2) if celdas else 0.0,
    }])


def comparar(*resumenes):
    """Concatena varios `resumen_tabla` para verlos lado a lado."""
    return pd.concat(resumenes, ignore_index=True)


def perfil_columnas(df):
    """Tipo, no nulos, faltantes y valores distintos por columna."""
    return pd.DataFrame({
        'tipo': df.dtypes.astype(str), 'no_nulos': df.notna().sum(), 'faltantes': df.isna().sum(),
        'pct_faltante': (df.isna().mean() * 100).round(2), 'distintos': df.nunique(dropna=True),
    }).rename_axis('columna')


def guardar_tabla(df, nombre, carpeta=None):
    """Escribe una tabla derivada como CSV UTF-8 (con BOM para abrirla bien en Excel)."""
    carpeta = carpeta or DATA_INTERIM
    carpeta.mkdir(parents=True, exist_ok=True)
    ruta = carpeta / f'{nombre}.csv'
    df.to_csv(ruta, index=False, encoding='utf-8-sig')
    return ruta
