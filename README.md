# Proyecto final de analítica: parejas ideales de pádel

Encontrar las parejas que maximizan la probabilidad de ganar el torneo según su estilo de juego. Se parte de tres fuentes: datos de video (pose y posición por frame), la traza de partidos y el formulario de jugadores.

**¿Por dónde empezar?** El trabajo completo (código, resultados y reporte de decisiones, por fases) está en un solo notebook: [notebooks/Proyecto_Parejas_Padel.ipynb](notebooks/Proyecto_Parejas_Padel.ipynb).

## Fases

1. **Manejo de datos** (actual): entendimiento de las fuentes, identidad de los jugadores, limpieza, imputación y variables de pareja.
2. **Predictiva:** regresión logística regularizada y Random Forest sobre el ganador del punto.
3. **Distribuciones:** distribución empírica por jugador y por variable aleatoria.
4. **Simulación:** combinaciones de parejas evaluadas con el modelo de la fase 2.

## Archivos del repositorio

```
README.md                         Este archivo: qué es cada cosa y cómo empezar
CLAUDE.md                         Reglas, contexto y registro de decisiones del proyecto
requirements.txt                  Librerías de Python necesarias
.gitignore                        Lo que no se sube a git (datos, cachés, entornos)
notebooks/
  Proyecto_Parejas_Padel.ipynb    Notebook único de trabajo: todas las fases y el reporte
src/                              Funciones del pipeline que usa el notebook
  __init__.py                     Marca src/ como paquete y lista sus módulos
  datos.py                        Rutas, constantes de cancha, lectura de fuentes, utilidades
  identidades.py                  Nombres canónicos, alias y códigos de estudiante
  fuentes.py                      Limpieza de la traza y del formulario
  video.py                        Calidad del video e identidad de cada jugador en él
docs/
  diccionario.md                  Diccionario de todas las columnas (fuentes y tablas derivadas)
  pose_keypoints_map.jpg          Imagen de los puntos de pose que usa el diccionario
data/                             (no se sube a git)
  raw/                            Los 3 archivos de entrada, sin modificar
  interim/                        Tablas intermedias que genera el notebook
  processed/                      Tablas finales para el modelo (aún vacía)
```

### Qué es y qué hace cada archivo

| Archivo | Qué es | Qué hace / cuándo usarlo |
|---|---|---|
| `notebooks/Proyecto_Parejas_Padel.ipynb` | El documento de trabajo del proyecto. | Se ejecuta de principio a fin (~2 min). Cada sección tiene Objetivo → Decisiones → Código → Resultado. La Fase 1 tiene las secciones 1.1 a 1.10 (la 1.2 reúne el entendimiento de los datos que antes estaba en dos notebooks aparte); las fases 2 a 4 están planificadas. Los anexos traen el registro de decisiones (A), los pendientes (B) y las tablas generadas (C). |
| `CLAUDE.md` | Las reglas del proyecto en un solo lugar: objetivo, fuentes, calidad del video por partido, reglas de identidad y limpieza, plan de variables y modelos, prácticas de código, decisiones `D#` y supuestos sin confirmar. | Consultarlo antes de tomar cualquier decisión de datos. Lo lee también el asistente de IA (Claude Code), así que cada decisión nueva se anota ahí. |
| `src/datos.py` | La base del pipeline. | Define las rutas de `data/`, la semilla (2026) y las medidas de la cancha. Lee las tres fuentes (`leer_video`, `leer_traza`, `leer_formulario`, `iterar_video` para recorrer el CSV grande por bloques) y trae las utilidades de reporte (`resumen_tabla`, `comparar`, `perfil_columnas`) y de escritura (`guardar_tabla`). |
| `src/identidades.py` | La tabla de equivalencias de jugadores. | Lleva cualquier nombre de cualquier fuente a su nombre canónico (completo) y a su código de estudiante de 6 dígitos. Aquí se agregan los alias nuevos, los jugadores sin formulario y las correcciones de código. |
| `src/fuentes.py` | La limpieza de las dos fuentes pequeñas. | Traza: controles de consistencia, corrección de erratas, códigos de los jugadores y formato largo jugador × punto. Formulario: selección de variables, escalas ordinales, imputación de los 3 jugadores sin formulario e IMC. |
| `src/video.py` | El tratamiento del CSV de video. | Detecta los problemas de calidad de cada punto (duplicados, `y` inválida, sin pose, homografía desplazada, fps raro) y asigna a cada jugador del video su código según la traza. Aquí irá también la limpieza frame × jugador (sección 1.8). |
| `src/__init__.py` | Archivo técnico de Python. | Permite hacer `from src import ...` desde el notebook. |
| `docs/diccionario.md` | El diccionario de datos. | Parte A: cada columna de las tres fuentes crudas, con su unidad, cómo se genera y si se usa, se deriva o se descarta. Parte B: cada columna de las tablas que genera el notebook. |
| `docs/pose_keypoints_map.jpg` | Imagen de referencia. | Muestra los puntos del cuerpo (hombro, codo, muñeca, raqueta, cabeza) que detecta el modelo de pose. La cita el diccionario. |
| `requirements.txt` | Lista de librerías. | `pip install -r requirements.txt` instala pandas, numpy, scipy, scikit-learn, matplotlib, openpyxl y jupyter. |
| `.gitignore` | Configuración de git. | Evita subir los datos (por tamaño y por datos personales), los cachés de Python y los entornos virtuales. |

### Tablas que genera el notebook (`data/interim/`)

| Tabla | Una fila por | Contenido |
|---|---|---|
| `equivalencias_nombres.csv` | nombre de cada fuente | Nombre original → nombre canónico → código |
| `jugadores.csv` | jugador (32) | Variables fijas del formulario, codificadas e imputadas, más el IMC |
| `participaciones.csv` | jugador × punto (1.132) | Equipo, fondo, lado, compañero, si ganó el punto y su track de video |
| `identidades_video.csv` | track de video (1.201) | A qué jugador corresponde cada track, cómo se asignó y si se conserva |
| `puntos.csv` | punto (287) | Ganador, jugadores, estado del video y banderas de calidad |

El detalle de cada columna está en la parte B de [docs/diccionario.md](docs/diccionario.md).

## Datos

Los datos no se suben al repositorio por su tamaño (el CSV de video pesa ~650 MB) y porque el formulario tiene datos personales. Cada integrante copia en `data/raw/` estos tres archivos, con estos nombres exactos:

- `Base_Videos_Final_202602.csv`: video consolidado (frame × jugador) de los 27 partidos con traza.
- `Traza_Partidos_202602.xlsx`: traza de partidos (hojas `CANCHA_1` y `CANCHA_2`).
- `Datos_Jugadores_Padel_202602.xlsx`: formulario de jugadores.

Luego se ejecuta el notebook de principio a fin; las tablas derivadas quedan en `data/interim/`.

## Instalación

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
source .venv/bin/activate     # macOS / Linux
pip install -r requirements.txt
```

## Origen

El proyecto parte del material del profesor (`padel-vision-analytics-light`). De ese material solo se conservó lo necesario para la analítica: la documentación de las columnas del CSV de video (reescrita en `docs/diccionario.md`), el mapa de keypoints y las dimensiones de la cancha. No se usa el pipeline de visión (YOLO, pose, tracking) porque los CSV de video se reciben ya generados.
