"""Identidad de los jugadores: nombres canónicos, alias y códigos de estudiante.

La llave de todo el proyecto es `codigo` (CODIGO_ESTUDIANTE como texto de 6 dígitos).
Los nombres solo se usan para llegar al código mediante la tabla de equivalencias.
"""

import re
import unicodedata

import pandas as pd

# Nombre canónico (completo, con dos apellidos cuando alguna fuente lo tiene) -> alias conocidos.
# Los alias se escriben normalizados (sin tildes, minúsculas, espacios simples). El propio nombre
# canónico normalizado siempre es alias de sí mismo. Decisión D11.
NOMBRES_CANONICOS = {
    'Alejandra Briget Deza Céspedes': [],
    'Ana Catalina Suarez Caballero': [],
    'Ana Sofía Santander Prieto': ['sofia santander'],
    'Andrés Felipe Santafé Buitrago': [],
    'Angel Santiago Toribio Godoy': [],
    'Angie Vanessa González Farfan': ['angie vanessa gonzalez 0', 'angie vanessa gonzalez'],
    'Antonio Benitez Rueda': [],
    'David Jeronimo Rojas Gutierrez': [],
    'David Neisa Tello': [],
    'Gabriel Armando González Sosa': [],
    'Gabriel López Acosta': [],
    'Isabella Arenas González': ['isabella arenas'],
    'Jacobo Figueroa Algarra': [],
    'Jeronimo Coy': [],
    'Jorge García': [],
    'Jose Fabian Garcia Camargo': ['jose fabian garcia'],
    'Juan Andrés Ordóñez Vanegas': ['juan andres ordonez'],
    'Juan Camilo Sánchez': [],
    'Juan Pablo Vanegas Herrera': [],
    'Juliana Gómez': [],
    'Karol Valentina Carranza Gantiva': [],
    'Laura Natalia García Camargo': [],
    'Laura Sofía Rodríguez González': [],
    'Luis Felipe Cuellar Santamaria': [],
    'Marcela Velásquez': [],
    'Mariana Alfonso Forero': [],
    'Miguel Delgado': [],
    'Miguel Medina Velez': ['miguel medina'],
    'Nikol Sofía Avendaño Suárez': ['nikol avendano'],
    'Samuel Alejandro Sanabria Ospina': [],
    'Santiago Pujol Zabala': [],
    'Valentina Rodríguez Zuluaga': ['valentina rodriguez'],
}

# Jugadores que juegan pero no llenaron el formulario: código entregado por el equipo (D9).
JUGADORES_SIN_FORMULARIO = {
    'Alejandra Briget Deza Céspedes': {'codigo': '407710', 'genero': 'Femenino'},
    'Juan Pablo Vanegas Herrera': {'codigo': '356629', 'genero': 'Masculino'},
    'Laura Sofía Rodríguez González': {'codigo': '324764', 'genero': 'Femenino'},
}

# Códigos del formulario que no tienen 6 dígitos (D10).
CORRECCIONES_CODIGO = {
    '99999': '099999',          # Jorge García: se antepone un 0.
    '1075684546': '107568',     # José Fabián García Camargo: se conservan los primeros 6 dígitos.
}

LONGITUD_CODIGO = 6
_PATRON_GENERICO = re.compile(r'^(player_\d+|jugador \d+)$')


def normalizar_nombre(valor):
    """Quita tildes, pasa a minúsculas y colapsa espacios. Devuelve '' si el valor falta."""
    if pd.isna(valor):
        return ''
    texto = unicodedata.normalize('NFKD', str(valor)).encode('ascii', 'ignore').decode('ascii')
    return re.sub(r'\s+', ' ', texto).strip().casefold()


def es_generico(nombre):
    """True para etiquetas de tracking sin identidad (`Player_17`, `Jugador 3`)."""
    return bool(_PATRON_GENERICO.match(normalizar_nombre(nombre)))


def mapa_alias():
    """Diccionario alias normalizado -> nombre canónico."""
    mapa = {}
    for canonico, alias in NOMBRES_CANONICOS.items():
        for a in [normalizar_nombre(canonico), *alias]:
            if a in mapa and mapa[a] != canonico:
                raise ValueError(f'Alias ambiguo: {a!r} -> {mapa[a]!r} y {canonico!r}')
            mapa[a] = canonico
    return mapa


def nombre_canonico(valor, mapa=None):
    """Nombre canónico de un nombre de cualquier fuente, o None si no está en la tabla."""
    mapa = mapa or mapa_alias()
    return mapa.get(normalizar_nombre(valor))


def normalizar_codigo(codigo):
    """CODIGO_ESTUDIANTE como texto de 6 dígitos, aplicando las correcciones registradas."""
    if pd.isna(codigo):
        return None
    texto = str(codigo).strip()
    if texto.endswith('.0'):
        texto = texto[:-2]
    texto = CORRECCIONES_CODIGO.get(texto, texto)
    if not (texto.isdigit() and len(texto) == LONGITUD_CODIGO):
        raise ValueError(f'Código sin corrección registrada o con longitud distinta de 6: {codigo!r}')
    return texto


def tabla_codigos(formulario):
    """Nombre canónico -> código, a partir del formulario más los jugadores sin formulario."""
    mapa = mapa_alias()
    filas = []
    for _, r in formulario.iterrows():
        filas.append({'nombre': nombre_canonico(r['NOMBRE'], mapa), 'codigo': normalizar_codigo(r['CODIGO_ESTUDIANTE']),
                      'origen_codigo': 'formulario'})
    for nombre, datos in JUGADORES_SIN_FORMULARIO.items():
        filas.append({'nombre': nombre, 'codigo': datos['codigo'], 'origen_codigo': 'equipo (D9)'})
    tabla = pd.DataFrame(filas)
    if tabla['nombre'].isna().any():
        raise ValueError('Hay nombres del formulario sin entrada en NOMBRES_CANONICOS.')
    if tabla['nombre'].duplicated().any() or tabla['codigo'].duplicated().any():
        raise ValueError('Nombre o código repetido en la tabla de códigos.')
    return tabla


def tabla_equivalencias(nombres_por_fuente, codigos):
    """Una fila por (fuente, nombre original) con su nombre canónico y código.

    nombres_por_fuente: {'formulario': iterable, 'traza': iterable, 'video': iterable}.
    Los nombres genéricos del tracking (`Player_N`, `Jugador N`) no entran.
    """
    mapa = mapa_alias()
    filas = []
    for fuente, nombres in nombres_por_fuente.items():
        for original in sorted({str(n).strip() for n in nombres if pd.notna(n)}):
            if es_generico(original):
                continue
            canonico = nombre_canonico(original, mapa)
            filas.append({'fuente': fuente, 'nombre_original': original, 'nombre': canonico,
                          'tipo_cruce': ('sin cruzar' if canonico is None else
                                         'exacto' if normalizar_nombre(original) == normalizar_nombre(canonico) else 'alias')})
    eq = pd.DataFrame(filas).merge(codigos[['nombre', 'codigo']], on='nombre', how='left')
    return eq.sort_values(['nombre', 'fuente', 'nombre_original'], na_position='first').reset_index(drop=True)


def sin_cruzar(equivalencias):
    """Nombres sin nombre canónico o sin código: se listan para buscarlos en Teams."""
    return equivalencias[equivalencias['nombre'].isna() | equivalencias['codigo'].isna()]
