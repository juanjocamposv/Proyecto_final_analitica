# CLAUDE.md — Pádel: parejas ideales

Responde en español, de forma directa y técnica.

## Objetivo

Encontrar las parejas de pádel que maximizan la probabilidad de ganar el torneo según su estilo de juego.

## Fases

1. **Manejo de datos (fase actual):** limpieza, imputación, granularidad y variables artificiales (diferencias, razones, sumas, combinaciones). Hechas: fuentes, identidad de jugadores, traza, formulario, diagnóstico e identidad del video (secciones 1.1–1.7 del notebook). Pendientes: limpieza frame × jugador y variables (1.8–1.10).
2. **Predictiva:** modelo supervisado que predice el ganador del punto.
3. **Distribuciones:** una distribución empírica por variable aleatoria y por jugador. Los valores fijos (altura, peso, nivel) no se modelan como distribución; entran como atributos.
4. **Simulación:** combinaciones de parejas y partidos muestreando las distribuciones de la fase 3, evaluadas con el modelo de la fase 2.

**Notebook de trabajo (único):** [notebooks/Proyecto_Parejas_Padel.ipynb](notebooks/Proyecto_Parejas_Padel.ipynb). Está organizado por fases y secciones; cada sección tiene Objetivo → Decisiones (D#) → Código → Resultado.

- **Fase 1:** 1.1 Fuentes; 1.2 Entendimiento de los datos (integra los antiguos notebooks `Entendimiento_*`, en las subsecciones 1.2.1–1.2.7: estructura del video, perfil completo, distribuciones, traza y formulario, relación entre fuentes, preguntas iniciales con su respuesta y resumen de hallazgos); 1.3 Identidad; 1.4 Traza; 1.5 Formulario; 1.6 Calidad del video; 1.7 Identidad de los tracks; 1.8–1.10 pendientes.
- **Fases 2 a 4:** planificadas, sin implementar.
- **Anexos:** A, registro de decisiones; B, supuestos y pendientes; C, tablas derivadas.
- **Cómo se trabaja:** todo avance nuevo se agrega en su sección, siguiendo el mismo formato, y su decisión se registra aquí y en el Anexo A. No se crean notebooks nuevos.
- **Edición y verificación:** el notebook se edita directamente y se vuelve a ejecutar completo (por ejemplo con `nbclient`) para verificar que corre sin errores antes de reportar números. Tarda unos 2 minutos.

## Fuentes de datos (`data/raw/`)

| Fuente | Archivo | Granularidad | Contenido |
|---|---|---|---|
| Video | `Base_Videos_Final_202602.csv` (~650 MB) | frame × jugador | Un solo CSV con todos los puntos: 718.936 filas, 74 columnas (las 67 del CSV por punto más `partido`, `punto`, `nombre_norm`, `schema_version` y 3 `*_implausible`). 27 partidos y 272 puntos. El fps varía por video (`duration` = 1/fps). |
| Traza | `Traza_Partidos_202602.xlsx`, hojas `CANCHA_1` y `CANCHA_2` | punto | 27 partidos y 283 puntos (cancha 1: 16 partidos y 164 puntos; cancha 2: 11 y 119), partidos a 7 puntos. Ganador (`PUNTO_EQUIPO_1`), marcador, parejas, bandas, LEFT/RIGHT y TOP/BOTTOM. La hoja `Original` programa 39 partidos; 12 no tienen traza ni video. `TDs` y `Resultados` son agregados que no se usan. |
| Formulario | `Datos_Jugadores_Padel_202602.xlsx`, hoja `Respuestas de formulario 1` | jugador | 29 filas con datos fijos. `Hoja 1` es un agregado de resultados que no se usa. |

- **Llave de unión:** `(partido, punto, codigo)`. `codigo` es `CODIGO_ESTUDIANTE` como texto de 6 dígitos. Ningún nombre se usa como llave: todo nombre pasa por la tabla de equivalencias (`src/identidades.py`) para llegar a su código.
- **Identidad en el video:** la da `data/interim/identidades_video.csv`, que tiene una fila por track `(partido, punto, player_id)` con su `codigo` y la marca `conservar` (D12).
- **Cobertura:** de los 283 puntos de la traza, 267 tienen video utilizable. 244 tienen a los 4 jugadores identificados, 16 solo al equipo BOTTOM (partidos 8 y 34) y 7 tienen 3 jugadores.
- **Columnas:** el detalle está en [docs/diccionario.md](docs/diccionario.md). La parte A cubre las fuentes crudas (unidades, cómo se generan y uso de cada columna) y debe consultarse antes de usar una columna. La parte B cubre las tablas derivadas de `data/interim/`.
- **Coordenadas del video:** en metros, con `x` ∈ [0, 10], `y` ∈ [0, 20] y la red en `y = 10`. Convención verificada con la traza (D13): BOTTOM en `y > 10` y, para ambos equipos, RIGHT en `x > 5`. Las constantes están en la sección 2 de `src/datos.py`.

### Calidad del video por partido (detección automática en `src/video.py`)

| Partido | Problema | Tratamiento |
|---|---|---|
| 5 | El punto 10 es una copia exacta del punto 9 | El video se conserva como punto 9; el punto 10 queda sin video (D7) |
| 8 | Tenía los nombres del partido 34; solo se detecta la pareja BOTTOM y `team` está vacío | Reasignado por posición (D12): Angel Toribio y Mariana Alfonso. El equipo 2 no tiene video |
| 17 | Etiquetas `Jugador 1–4`, `y` = copia de `x` y sin pose | Reasignado por posición (`team` + `x`). Solo sirven variables de `x` y de tiempo (por confirmar) |
| 27 | El equipo top aparece en `y` ≈ 15 (homografía desplazada); `team` sí es correcto | No usar su profundidad absoluta (por confirmar) |
| 32 | 300 frames por punto con fps de 11,8 a 39,9 | Calcular el fps por punto (D2) |
| 34 | La etiqueta «David Neisa» es en realidad Alejandra Deza; solo se detecta bien la pareja BOTTOM | Reasignado por posición (D12) |
| 7-11, 11-12, 23-9, 32-13 | Puntos de video sin traza (el partido ya había terminado) | Excluidos (D15) |

## Repositorio

- `data/raw/`: datos de entrada sin modificar. `data/interim/` y `data/processed/`: salidas. Ninguna carpeta de datos se versiona; cada integrante copia los tres archivos crudos en `data/raw/`.
- `src/`: funciones reutilizables del pipeline, en 4 módulos (D17). El notebook los importa como `from src import datos, fuentes`, `identidades as idn` y `video as vid`.
  - `datos.py`: rutas, semilla, constantes de la cancha, lectura de las fuentes (`leer_*`, `iterar_video`) y utilidades (`resumen_tabla`, `comparar`, `perfil_columnas`, `guardar_tabla`).
  - `identidades.py`: `NOMBRES_CANONICOS`, `JUGADORES_SIN_FORMULARIO`, `CORRECCIONES_CODIGO` y la tabla de equivalencias.
  - `fuentes.py`: traza (`validar_traza`, `limpiar_traza`, `tabla_puntos_traza`, `traza_larga`) y formulario (`limpiar_formulario`, `imputar_sin_formulario`, `agregar_derivadas_jugadores`, `ORDINALES`).
  - `video.py`: `diagnosticar_puntos`, `resumir_tracks`, `concordancia_por_partido`, `asignar_identidades`, `enlazar_participaciones` y `tabla_puntos`. La limpieza de la sección 1.8 también irá aquí.
  - Las fases siguientes agregan un módulo cada una (`modelo.py`, `simulacion.py`), no más archivos sueltos.
- `notebooks/`: solo `Proyecto_Parejas_Padel.ipynb`.
- `docs/`: `diccionario.md` (único diccionario) y `pose_keypoints_map.jpg` (mapa de keypoints que cita el diccionario).
- **Eliminados (D17):** `Entendimiento_Base_Videos_Final.ipynb` y `Entendimiento_Tres_CSV.ipynb` (integrados en la sección 1.2), `docs/guia_contexto_proyecto.pdf` (guía del 3-nov, superada por este archivo), `referencia/normalize_traza_profesor.py` (lo reemplaza la sección de traza de `src/fuentes.py`) y los dos diccionarios separados.
- No se usa el pipeline de visión del profesor (YOLO, pose, tracking): los CSV de video se reciben ya generados.

## Reglas de identidad (fase 1, secciones 1.3 y 1.7)

- **Nombres (D9):** las variantes de un nombre son el mismo jugador; no se separan en seguras y dudosas. El nombre canónico es el completo, con dos apellidos cuando alguna fuente lo tiene. Alias y nombres canónicos están en `NOMBRES_CANONICOS` de `src/identidades.py`.
- **Códigos (D8):** todos tienen 6 dígitos y se guardan como texto. Las correcciones están en `CORRECCIONES_CODIGO`.
- **Tracks del video (D12):** se usa la etiqueta del video si el partido tiene ≥50 % de filas con nombres de la alineación y ≥80 % de concordancia con la casilla (fondo, lado) de la traza. Si no, se asigna por posición contra la traza (fondo por `team`, o por `y` si `team` falta; lado por `x`). Los `Player_N` y los tracks sin casilla compatible se descartan (D14).
- **Búsqueda en Teams (proceso recurrente):** cada vez que un jugador no cruza con la tabla de equivalencias (con nuevos videos, partidos o formularios), la sección 1.3 lo lista y se pide al equipo que lo busque en Teams. Con el nombre y el código confirmados, se agrega a `NOMBRES_CANONICOS` (y a `JUGADORES_SIN_FORMULARIO` si no tiene formulario) y se vuelve a correr la fase 1. Hoy no hay jugadores sin cruzar.

## Reglas de limpieza del video (sección 1.8, pendiente)

- **Pelota:** descartar todas sus columnas. Las posiciones están interpoladas y `ball_speed_mps` tiene mediana 0 y máximos de 894 m/s. Los golpes se aproximan con picos de velocidad de raqueta. **Ojo:** `player_hits_ball` no siempre es False en la base consolidada: es verdadero en 2.264 filas (0,3 %) de 26 partidos. Su uso está pendiente de decisión del equipo; mientras tanto se descarta.
- **Píxeles:** dividir toda medida en píxeles por `body_scale`, que depende de la distancia a la cámara (mediana por track: ~32 px en TOP y ~77 px en BOTTOM). Los ángulos y las variables `*_norm` ya son comparables.
- **Velocidad y aceleración:** las posiciones están cuantizadas (~5 cm), así que no se usan las columnas originales. Suavizar la trayectoria y recalcular con ventanas de 0,5 a 1 s, con el fps de cada punto.
- **Pose:** interpolar solo huecos de ≤5 frames; el resto queda NaN. Guardar `pose_coverage` por jugador y punto (varía de 0 % a 100 % según el partido).
- **Faltantes menores:** `team` se completa con el valor constante del jugador y la distancia al compañero se recalcula desde las posiciones.
- **Rol:** derivarlo de la posición; `zone` no es un rol real.
- **Fuera de la cancha:** hay posiciones con `x` hasta 11,8 y `y` de −0,5 a 20,5. Recortar o marcar.
- **Descartadas:** `duration` (solo para calcular el fps), `total_frames`, `video_duration`, `prev_*`, las coordenadas en píxeles (solo para derivar), `upper_arm_length`, `forearm_length`, `arm_extension`, `racket_vector_x/y`, las orientaciones de antebrazo y brazo, `schema_version` y `*_implausible` (solo existen en los partidos 10, 13 y 39).
- **Solo se limpian los tracks con `conservar = True`** en `identidades_video.csv`.

## Reglas del formulario y la traza

- **Fuga de datos:** `Puntos_Total`, `Partidos_Jugados` y `Puntos_Por_Partido` nunca son predictoras; solo sirven para validar. `Puntos_Total` coincide con la traza en 27 de 27 jugadores.
- **Descartadas:** ciudad, programa, semestre, razón y objetivo de participación, interés y clases de pádel, desayuno, género musical, comentarios y `RITMO_CARDIACO`.
- **Usables:** `ESTATURA`, `TALLA` (peso), `MANO_DOMINANTE` (solo 2 zurdos), `NIVEL_ACTUAL_PADEL`, `TIEMPO_JUGANDO_PADEL`, `EXPERIENCIA_PADEL`, `PRACTICA_OTRO_DEPORTE_RAQUETA`, `ESTADO_FISICO`, `FRECUENCIA_DEPORTE`, `EDAD`, `GENERO` y `LESIONES_LY`. Las escalas ordinales están en `ORDINALES` de `src/fuentes.py` (D11). La tabla de salida es `data/interim/jugadores.csv` (32 jugadores, 3 imputados).
- **Lado de cancha:** el equipo 1 siempre está en BOTTOM (en ambas canchas). Incluirlo como variable de control, sin sacar conclusiones fuertes.
- **Traza:** pasa todos los controles (un ganador por punto, marcador igual al acumulado, puntos consecutivos, lados fijos por partido). El equipo 1 gana el 43,1 % de los puntos.

## Variables

Granularidad: **frame × jugador → jugador × punto → pareja × punto** (con la pareja rival como contexto).

- **Jugador × punto:** profundidad media y desviación, % cerca de la red, lado, área cubierta; distancia recorrida, % quieto, ráfagas rápidas, velocidad p90; % raqueta sobre el hombro, proporción por dirección de raqueta, variabilidad de ángulos; velocidad de raqueta normalizada en p90/p95 (no el máximo) y conteo de picos; `pose_coverage`.
- **Pareja × punto:** diferencias (complementariedad), razones min/max (balance), sumas (capacidad total), cohesión (media y desviación de `distance_player_to_teammate_m`), sincronía (correlación de movimiento entre compañeros), fijas de pareja (estatura, nivel, experiencia, género, manos y si cada jugador está en el lado de su mano), relativas al rival (A − B), contexto (lado, cobertura de pose).
- **Meta:** 10 a 15 variables finales. Revisar correlación, redundancia, varianza baja y faltantes antes de elegir.

## Modelado

- **Objetivo:** `PUNTO_EQUIPO_1` por punto. La probabilidad de ganar el partido a 7 puntos se deriva de p(ganar el punto).
- **Validación:** dejar un partido fuera (27 partidos; los puntos de un partido no son independientes). Nunca mezclar puntos del mismo partido entre entrenamiento y prueba.
- **Modelos:** regresión logística regularizada (referencia) contra Random Forest.
- **Métricas:** AUC, log-loss, calibración e importancia de variables.
- **Distribuciones (fase 3):** comparar el bootstrap empírico contra una paramétrica ajustada, considerando las pocas observaciones por jugador.
- **Simulación (fase 4):** ranking de parejas por probabilidad media de ganar contra el resto, con intervalos de incertidumbre.
- Las parejas casi no se repiten (30 jugadores en la traza): el modelo aprende de características de pareja, no de identidades.

## Prácticas de código

- Python con pandas, numpy y scikit-learn.
- Ejecutar el código y verificar resultados; no reportar números sin correrlo.
- Pipeline como funciones reutilizables para cualquier `(partido, punto)`, no código específico de un partido. Los casos especiales se detectan con reglas (banderas de `diagnosticar_puntos`), no con listas fijas.
- Mostrar un resumen antes y después de cada paso de limpieza (`datos.resumen_tabla` y `datos.comparar`: filas, faltantes y rangos).
- No modificar los datos crudos; escribir los resultados en archivos nuevos (`datos.guardar_tabla`).
- Entregar cada tabla derivada con su diccionario de variables (parte B de `docs/diccionario.md`).
- Semilla única: `datos.SEMILLA` = 2026.
- Gráficos en matplotlib: una sola serie en `#2a78d6`, sin leyenda cuando hay una serie, cuadrícula tenue y bordes superior y derecho ocultos (mismo estilo que la sección 1.2).
- **Mantener el repositorio simple (D17):** antes de crear un archivo nuevo, ver si cabe en uno existente (notebook, módulo de `src/`, `docs/diccionario.md`, este archivo o el README). Si cambia la estructura, actualizar el README, que explica qué es y qué hace cada archivo.

## Registro de decisiones

Ordenado por sección del notebook. La misma tabla está en el Anexo A del notebook.

| # | Sección | Decisión | Fuente |
|---|---|---|---|
| D17 | General | Repositorio simplificado: un solo notebook (los dos de entendimiento integrados en la sección 1.2), un solo diccionario (`docs/diccionario.md`) y `src/` en 4 módulos (`datos`, `identidades`, `fuentes`, `video`). Se eliminaron la guía PDF del 3-nov y el script del profesor para la traza. | Equipo (2026-11-07) |
| D16 | 1.1 | Los tres archivos crudos van en `data/raw/`; las salidas en `data/interim/` y `data/processed/`. | Equipo (2026-11-06) |
| D4 | 1.3 / 1.5 | Jugadores sin formulario con código entregado por el equipo: Alejandra Briget Deza Céspedes (407710), Juan Pablo Vanegas Herrera (356629) y Laura Sofía Rodríguez González (324764). Se imputan con datos de su sexo: continuas (edad, estatura, peso) con la media, más ruido N(0, sd de las mujeres) para las dos mujeres, con semilla fija y recorte al rango observado; ordinales con la mediana (la categoría inferior si cae entre dos); categóricas (mano, lesiones, otro deporte de raqueta) con la moda. Llevan `imputado = True`. | Equipo (2026-11-06) |
| D8 | 1.3 | Todos los códigos tienen 6 dígitos y se guardan como texto. Jorge García: 99999 → `099999`. José Fabián García Camargo: 1075684546 → `107568` (primeros 6 dígitos). | Equipo (2026-11-06) |
| D9 | 1.3 | Las variantes de nombre son el mismo jugador; se usa el nombre completo con dos apellidos. No se separan seguras y dudosas. | Equipo (2026-11-06) |
| D3 | 1.4 | Variable objetivo: ganador del punto (`PUNTO_EQUIPO_1`). | Equipo (2026-11-06) |
| D13 | 1.4 / 1.7 | `RIGTH` → `RIGHT` en la traza. Convención: BOTTOM en `y > 10` y RIGHT en `x > 5` para ambos equipos. | Verificado con los datos (99 % de concordancia) |
| D1 | 1.5 | `TALLA` es el peso en kg. Se calcula IMC = `TALLA` / (`ESTATURA` en m)². | Equipo (2026-11-06) |
| D5 | 1.5 | La fila sin nombre (152 cm, 48 kg) es Ana Catalina Suárez Caballero (337448); ya viene resuelta en el formulario actual. | Formulario actual (2026-11-06) |
| D10 | 1.5 | `FRECUENCIA_DEPORTE`: «0 veces» y «No practico deportes» se unen en el nivel 0. | **Por confirmar** |
| D11 | 1.5 | Escalas ordinales (0 = menor) de nivel, tiempo jugando, experiencia, estado físico y frecuencia. `TIEMPO_JUGANDO_PADEL` es categórica, así que se imputa como ordinal (mediana). | Análisis (ajuste a D4) |
| D2 | 1.6 / 1.8 | `racket_speed`, `wrist_speed` y `elbow_speed` en px/s; velocidades angulares en °/s. El fps se obtiene por punto como 1/`duration`; no se asume 60. | Código del pipeline del profesor (ver la parte A de `docs/diccionario.md`) |
| D6 | 1.6 | Los CSV de video se reciben ya generados; del material del profesor solo se conserva lo necesario para la analítica. Su base normalizada no se usa: es de otro torneo. | Revisión del repositorio del profesor (2026-11-06) |
| D7 | 1.6 | El partido 5, punto 10, es una copia exacta del punto 9: el video se conserva como punto 9 y el punto 10 queda sin video. Regla general: conservar el de menor número y revisar cada duplicado nuevo con el equipo. | Equipo (2026-11-06) |
| D12 | 1.7 | Identidad de los tracks: por etiqueta si es coherente con la traza (≥50 % de filas en la alineación y ≥80 % de concordancia de casilla); si no, por posición contra la traza (asignación codiciosa por filas × p(fondo) × p(lado)). Se aplicó a los partidos 8, 17 y 34. | Equipo: «asignar según traza» (2026-11-06) |
| D14 | 1.7 | Los tracks `Player_N` y los que quedan sin casilla compatible se descartan de la identidad. | Análisis |
| D15 | 1.7 | Se excluyen los puntos de video sin traza (7-11, 11-12, 23-9 y 32-13). | Análisis (con supuesto de numeración) |

## Supuestos sin confirmar

Si un dato falta o un supuesto no está confirmado, decirlo y preguntar antes de asumirlo.

- D10: unir «0 veces» y «No practico deportes» en `FRECUENCIA_DEPORTE`.
- `player_hits_ball`: es verdadero en el 0,3 % de las filas (26 de 27 partidos). Falta decidir si se usa para contar golpes o si se mantienen los picos de velocidad de raqueta.
- La numeración `punto` del video es la misma que `ID_PUNTO` de la traza. Hay 4 videos con un punto extra al final del partido.
- Tratamiento de los partidos 17 (sin `y` ni pose), 27 (`y` del equipo top desplazada) y 8 y 34 (sin video del equipo 2): se decide en las secciones 1.8 y 1.10.
- Faltan los videos de 15 puntos de la traza. 12 partidos programados (2, 4, 6, 12, 15, 19, 21, 22, 30, 33, 36, 38) no tienen traza ni video; `Partidos_Jugados` los cuenta, pero con 0 puntos.
- Cinco jugadores tienen un solo apellido en todas las fuentes (Jeronimo Coy, Jorge García, Juliana Gómez, Marcela Velásquez, Miguel Delgado). Buscar en Teams es opcional.

## Cierre de sesión

Al terminar, resumir en viñetas: qué se decidió, qué archivos se generaron y qué queda pendiente.
