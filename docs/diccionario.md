# Diccionario de datos

Describe cada columna que usa el proyecto: las tres fuentes crudas que llegan a `data/raw/` (parte A) y las tablas derivadas que genera el notebook [Proyecto_Padel.ipynb](../notebooks/Proyecto_Padel.ipynb) en `data/interim/` (parte B). Las reglas de datos y las decisiones `D#` están en el [README](../README.md#reglas-de-datos). La información del CSV de video sale de la documentación y el código del pipeline de visión del profesor (repositorio `padel-vision-analytics-light`), que no se incluye aquí.

- **Parte A — Fuentes crudas:** [A1. Video](#a1-csv-de-video-base_videos_final_202602csv) · [A2. Traza](#a2-traza-de-partidos-traza_partidos_202602xlsx) · [A3. Formulario](#a3-formulario-de-jugadores-datos_jugadores_padel_202602xlsx)
- **Parte B — Tablas derivadas:** `equivalencias_nombres`, `jugadores`, `participaciones`, `identidades_video`, `puntos`, `video_limpio`, `distribuciones_jugador`, `processed/tabla_modelo` y `processed/ranking_parejas`.

---

## Parte A — Fuentes crudas (`data/raw/`)

**Leyenda de uso:** **Usar** = entra a la agregación · **Derivar** = solo para calcular otra variable · **Recalcular** = no usar el valor original · **Descartar** = no se usa · **Fuera del plan** = existe pero el plan no la contempla.

### A1. CSV de video `Base_Videos_Final_202602.csv`

Una fila = un jugador en un frame. Un solo CSV consolidado con todos los puntos (718.936 filas, 27 partidos, 272 puntos), con 74 columnas: las 67 del CSV por punto (`Partido_X_Punto_Y_CON_NOMBRES.csv`) más las 7 de la tabla «Columnas agregadas en la base consolidada». En el repositorio se guarda como `data/raw/video/partido_XX.parquet` (un archivo por partido, mismo contenido; D5).

#### Cómo se generan (importa para la limpieza)

- **Sistema de coordenadas (m):** homografía de la imagen a la cancha. `x` ∈ [0, 10] a lo ancho, `y` ∈ [0, 20] a lo largo, (0, 0) = esquina del fondo lejano izquierdo. Red en `y = 10`. El equipo `top` está en `y < 10`.
- **fps:** se leen del video. `duration` = 1/fps; no es constante entre videos.
- **`team`:** se asigna en cada frame ordenando a los jugadores por `y` (los 2 de menor `y` = `top`). Si en el frame no hay exactamente 4 jugadores detectados, queda vacío. Eso explica los frames sin `team`.
- **`distance_player_to_teammate_m`:** se calcula con la misma agrupación por `y`; queda vacía cuando no se puede formar la pareja.
- **`distance_player_to_net_m`:** `|y − 10|`.
- **`zone`:** `back_court` si el jugador está a más de 6,95 m de la red; si no, `left_box`/`right_box` según el lado, con la referencia invertida entre `top` y `bottom`. Es geométrica, no un rol.
- **Pelota:** sus posiciones se interpolan linealmente en todos los huecos y se rellenan el inicio y el final del punto con la primera y la última detección. Por eso parecen relleno.
- **Velocidades:** son diferencias entre frames consecutivos divididas por `duration` (sin suavizado).
- **Pose:** un modelo YOLO11s-pose detecta solo el brazo de juego (hombro, codo, muñeca, punta de raqueta y cabeza; ver [pose_keypoints_map.jpg](pose_keypoints_map.jpg)) y se asocia al jugador por IoU ≥ 0,3. Los keypoints con confianza 0 quedan vacíos. Todo lo de pose está en píxeles.

#### Columnas

Notación de pose: S = hombro, E = codo, W = muñeca, R = punta de raqueta, H = cabeza. Las coordenadas `y` de imagen crecen hacia abajo.

| Columna | Significado | Unidad | Uso |
|---|---|---|---|
| `frame_idx` | Número de frame | — | Usar (orden temporal) |
| `duration` | Duración de un frame | s | Derivar (fps = 1/`duration`) |
| `total_frames`, `video_duration` | Totales del video | —, s | Descartar |
| `player_id` | ID del track dentro del punto. No es estable entre puntos y puede pasar de 4 cuando hay tracks extra (`Player_N`) | — | Usar (llave del track con `partido`, `punto`) |
| `player_name` | Nombre del jugador | — | Derivar (la identidad final sale de `identidades_video.csv`) |
| `team` | `top` / `bottom` | — | Usar (completar con el valor constante del jugador) |
| `zone` | `left_box`, `right_box`, `back_court` | — | Descartar como rol; el rol se deriva de la posición |
| `player_position_x`, `player_position_y` | Posición del jugador | m | Usar (suavizar) |
| `distance_player_to_net_m` | Distancia a la red | m | Recalcular desde la posición suavizada |
| `distance_player_to_teammate_m` | Distancia al compañero | m | Descartar |
| `player_displacement` | Desplazamiento entre frames | m | Descartar (la velocidad se recalcula desde la posición) |
| `player_speed_mps`, `player_acceleration_mps2` | Velocidad y aceleración | m/s, m/s² | Recalcular (ruido por cuantización) |
| `prev_x`, `prev_y`, `prev_speed` | Valores del frame anterior | — | Descartar |
| `ball_position_x`, `ball_position_y`, `ball_position_x_prev`, `ball_position_y_prev` | Posición de la pelota | m | Descartar |
| `ball_displacement`, `ball_speed_mps`, `distance_ball_to_net`, `distance_player_to_ball_m` | Movimiento y distancias de la pelota | m, m/s | Descartar |
| `player_hits_ball`, `time_since_last_hit` | Golpe detectado (pelota a ~0,15 m del jugador) y tiempo desde el último | bool, s | Descartar (D14: depende de la posición interpolada de la pelota; golpes por picos de `racket_speed`) |
| `head_x/y`, `shoulder_x/y`, `elbow_x/y`, `wrist_x/y`, `racket_tip_x/y` | Keypoints H, S, E, W, R | px | Descartar (D15) |
| `upper_arm_length`, `forearm_length`, `arm_extension` | dist(S,E), dist(E,W), dist(S,W) | px | Descartar (D15) |
| `racket_length_visible` | dist(W,R) | px | Descartar (D15) |
| `elbow_angle` | Ángulo (S, E, W) | ° | Descartar (D15) |
| `shoulder_angle` | Ángulo (E, S, H) | ° | Descartar (D15) |
| `racket_vector_x`, `racket_vector_y` | (R − W) unitario | — | Descartar (D15) |
| `racket_orientation_deg` | atan2(Ry − Wy, Rx − Wx) | ° | Descartar (D15) |
| `racket_direction` | Orientación en 8 direcciones | categórica | Descartar (D15) |
| `racket_height_relative_to_head` | Hy − Ry (> 0 = sobre la cabeza) | px | Descartar (D15) |
| `arm_extension_ratio` | dist(S,W) / (dist(S,E) + dist(E,W)) · 100 | % | Descartar (D15) |
| `forearm_orientation_deg`, `upper_arm_orientation_deg` | Orientación de antebrazo y brazo | ° | Descartar (D15) |
| `wrist_above_head` | Wy < Hy | bool | Descartar (D15) |
| `racket_above_shoulder` | Ry < Sy | bool | Descartar (D15) |
| `wrist_height_rel_shoulder` | Sy − Wy | px | Descartar (D15) |
| `wrist_height_norm` | `wrist_height_rel_shoulder` / `body_scale` | — | Descartar (D15) |
| `racket_height_norm` | `racket_height_relative_to_head` / `body_scale` | — | Descartar (D15) |
| `body_scale` | dist(H,S) | px | Descartar (D15) |
| `pose_confidence` | Confianza media de los keypoints | 0–1 | Descartar (D15) |
| `pose_available` | Hubo pose en el frame (texto `True`/`False`) | bool | Derivar (`pct_pose` por punto) |
| `racket_displacement` | dist(R_t, R_t−1) | px | Descartar (D15) |
| `racket_speed` | `racket_displacement` / `duration` | px/s | Descartar (D15) |
| `wrist_speed` | dist(W_t, W_t−1) / `duration` | px/s | Descartar (D15) |
| `racket_acceleration`, `elbow_speed` | Δ`racket_speed` / `duration`, dist(E_t, E_t−1) / `duration` | px/s², px/s | Descartar (D15) |
| `elbow_angular_velocity` | Δ`elbow_angle` / `duration` | °/s | Descartar (D15) |
| `racket_angular_velocity` | Δ`racket_orientation_deg` (ajustado a ±180) / `duration` | °/s | Descartar (D15) |

#### Columnas agregadas en la base consolidada

| Columna | Significado | Uso |
|---|---|---|
| `partido`, `punto` | Partido y punto del video; se cruzan con `ID_PARTIDO`, `ID_PUNTO` de la traza | Usar (llave) |
| `nombre_norm` | `player_name` sin tildes y en minúsculas. Hay etiquetas genéricas `player_N` y `jugador N` | Derivar (identidad, ver D17) |
| `schema_version` | Versión del esquema; solo vale 2 en los partidos 10, 13 y 39 (10 % de las filas) | Descartar |
| `player_speed_implausible`, `ball_speed_implausible`, `row_implausible` | Controles de plausibilidad; solo existen donde `schema_version` = 2 | Descartar |

### A2. Traza de partidos (`Traza_Partidos_202602.xlsx`)

Una fila = un punto. Hojas útiles: `CANCHA_1` (partidos impares, 164 puntos) y `CANCHA_2` (partidos pares, 119 puntos). La hoja `Original` programa los 39 partidos con parejas y bandas (12 no tienen traza); `TDs` y `Resultados` son tablas dinámicas y agregados que no se usan. Se lee con `datos.leer_traza` y se limpia con las funciones de traza de `src/fuentes.py`.

| Columna | Significado |
|---|---|
| `ID_PARTIDO` | Partido (pares en la cancha 2, impares en la 1) |
| `ID_PUNTO` | Número de punto dentro del partido |
| `ID_JUGADOR_{1,2}_EQUIPO_{1,2}` | **Nombre** del jugador (no es un código; se cruza con la tabla de equivalencias) |
| `MARCADOR` | Marcador acumulado tras el punto (`equipo1-equipo2`) |
| `Equipo{1,2}_Banda{1,2}` | Color de la banda del jugador |
| `POSICION_JUGADOR_{1,2}_EQUIPO_{1,2}` | `LEFT` / `RIGHT` (vista de la cámara: RIGHT es `x > 5` para ambos equipos). Contiene la errata `RIGTH` en el partido 34 |
| `EQUIPO_{1,2}_POSICION` | `TOP` / `BOTTOM` |
| `PUNTO_EQUIPO_1`, `PUNTO_EQUIPO_2` | 1 si el equipo ganó el punto. `PUNTO_EQUIPO_1` es la variable objetivo |

El formato largo jugador × punto es la tabla derivada `participaciones.csv` (parte B); su llave es el código de estudiante, no el nombre.

### A3. Formulario de jugadores (`Datos_Jugadores_Padel_202602.xlsx`)

Una fila = un jugador (29 filas), en la hoja `Respuestas de formulario 1`. La hoja `Hoja 1` es un agregado de resultados que no se usa. Las columnas que se usan, las que se descartan y las reglas de imputación están en el [README](../README.md#formulario-y-traza) (secciones «Formulario y traza» y «Registro de decisiones»). Las escalas ordinales están en `src/fuentes.py` (`ORDINALES`).

---

## Parte B — Tablas derivadas (`data/interim/`)

CSV UTF-8 con BOM, salvo `video_limpio`, que es Parquet por su tamaño. Ninguna se versiona; se regeneran ejecutando el notebook.

| Tabla | Grano | Filas | Sección |
|---|---|---|---|
| `equivalencias_nombres.csv` | fuente × nombre original | 89 | 1.2 |
| `jugadores.csv` | jugador | 32 | 1.2 |
| `participaciones.csv` | jugador × punto (de la traza) | 1.132 | 1.2 / 1.3 |
| `identidades_video.csv` | track de video `(partido, punto, player_id)` | 1.201 | 1.2 |
| `puntos.csv` | punto (unión externa traza–video) | 287 | 1.2 |
| `video_limpio.parquet` | frame × jugador (tracks conservados) | 705.888 | 1.2 |
| `distribuciones_jugador.csv` | jugador (distribuciones para la fase 4) | 30 | 3.3 |
| `processed/tabla_modelo.csv` | equipo × punto (tabla para el modelo) | 448 | 1.5 |
| `processed/ranking_parejas.csv` | pareja (resultado de la simulación) | 435 | 4.3–4.5 |

### `equivalencias_nombres.csv`

| Columna | Descripción |
|---|---|
| `fuente` | `formulario`, `traza` o `video` |
| `nombre_original` | Nombre tal como aparece en la fuente |
| `nombre` | Nombre canónico (completo, D11) |
| `tipo_cruce` | `exacto` (igual tras normalizar), `alias` (variante registrada) o `sin cruzar` (buscar en Teams) |
| `codigo` | Código de estudiante de 6 dígitos (texto) |

### `jugadores.csv`

| Columna | Tipo | Descripción |
|---|---|---|
| `codigo` | texto | Código de estudiante, 6 dígitos (D10). Llave |
| `nombre` | texto | Nombre canónico |
| `genero` | categórica | `Femenino` / `Masculino` |
| `mujer` | 0/1 | 1 si `genero` = Femenino |
| `edad` | entero | Años |
| `estatura_cm` | entero | Estatura en cm |
| `peso_kg` | real | `TALLA` del formulario = peso en kg (D8) |
| `imc` | real | peso / (estatura en m)² |
| `mano_dominante` | categórica | `Derecha` / `Izquierda` |
| `zurdo` | 0/1 | 1 si es zurdo (solo 2 jugadores) |
| `nivel_padel` | ordinal 0–3 | Primera vez, Principiante, Intermedio, Avanzado |
| `tiempo_jugando_padel` | ordinal 0–2 | <6 meses, 6 meses–1 año, >1 año |
| `experiencia_padel` | 0/1 | Tenía experiencia previa en pádel |
| `otro_deporte_raqueta` | categórica | `No`, `Tenis`, `Tenis de mesa`, `Squash`, `Otro` |
| `practica_otra_raqueta` | 0/1 | 1 si practica otro deporte de raqueta |
| `estado_fisico` | ordinal 0–3 | Malo, Regular, Bueno, Excelente |
| `frecuencia_deporte` | ordinal 0–3 | 0 veces o no practica (D12), 1–2, 3–4, >4 veces por semana |
| `lesiones_ly` | categórica | Lesiones en el último año (`Sí` / `No`) |
| `lesion_ultimo_anio` | 0/1 | 1 si `lesiones_ly` = Sí |
| `imputado` | bool | True para los 3 jugadores sin formulario (D9) |

### `participaciones.csv`

| Columna | Descripción |
|---|---|
| `partido`, `punto`, `cancha` | Identificación del punto |
| `equipo` | 1 o 2 |
| `codigo`, `nombre` | Jugador (canónico) |
| `nombre_traza` | Nombre tal como aparece en la traza |
| `fondo` | `TOP` / `BOTTOM` (el equipo 1 siempre es BOTTOM) |
| `lado` | `LEFT` / `RIGHT` (vista de la cámara, D13) |
| `banda` | Color de la banda |
| `codigo_companero` | Código del compañero de pareja |
| `gano_punto` | 1 si su equipo ganó el punto |
| `player_id` | Track del video asignado al jugador en ese punto (vacío si no hay) |
| `filas_video` | Frames del jugador en el video de ese punto (0 si no hay) |
| `metodo_identidad` | `etiqueta` o `posicion` (D17); vacío si no hay track |

Las 14 variables preliminares del jugador (1.3, D21). Las de video quedan vacías si el jugador no tiene track o si su track dura menos de 2 s (D21), y se miden respecto de su fondo (D22). Las del formulario se repiten en todos los puntos del jugador. En la tabla del modelo entran 12: `dist_red_m` y `pct_quieto` salen por correlación (D24).

| Columna | Grupo | Unidad / escala | Descripción |
|---|---|---|---|
| `nivel_padel_1`, `nivel_padel_2` | nivel de juego | 0/1 | One-hot del nivel (D21): 1 = principiante; 2 = intermedio o avanzado. Base (las dos en 0): primera vez |
| `tiempo_jugando_padel` | nivel de juego | 0/1 | 0 = menos de 6 meses, 1 = 6 meses o más |
| `experiencia_padel` | nivel de juego | 0/1 | 1 si ya había jugado pádel |
| `practica_otra_raqueta` | nivel de juego | 0/1 | 1 si practica otro deporte de raqueta |
| `atacante` | tipo de juego | 0/1 | 1 si pasa al menos la mitad del punto más cerca de la red que la mediana de distancia de su fondo (7,66 m en BOTTOM y 6,53 m en TOP) |
| `dist_red_m` | tipo de juego | m | Distancia media a la red menos la mediana de su fondo (negativa = más cerca de la red que el jugador típico de su fondo) |
| `vel_media_mps` | tipo de juego | m/s | Velocidad media menos la mediana de su fondo |
| `pct_quieto` | tipo de juego | % | % del tiempo con velocidad menor que la mediana de su fondo |
| `estatura_cm` | atributos físicos | cm | Estatura |
| `imc` | atributos físicos | kg/m² | Índice de masa corporal |
| `edad` | atributos físicos | años | Edad |
| `estado_fisico_1`, `estado_fisico_2` | atributos físicos | 0/1 | One-hot del estado físico (D21): 1 = bueno; 2 = excelente. Base: malo o regular |
| `frecuencia_deporte_1` a `_3` | atributos físicos | 0/1 | One-hot de las veces por semana que hace deporte (D12, D21): 1 = 1–2; 2 = 3–4; 3 = más de 4. Base: 0 veces |
| `mujer` | atributos físicos | 0/1 | 1 si es mujer |

### `identidades_video.csv`

| Columna | Descripción |
|---|---|
| `partido`, `punto`, `player_id` | Llave del track |
| `etiqueta_video` | `nombre_norm` original del track |
| `filas` | Frames del track |
| `x_mediana`, `y_mediana` | Mediana de la posición (m); `y` vacía si el punto tiene `y_invalida` |
| `fuente_fondo` | De dónde sale el fondo: `team`, `y` o `sin dato` |
| `p_bottom`, `p_right` | Proporción de frames en BOTTOM (`team` o `y > 10`) y en RIGHT (`x > 5`) |
| `fondo_pos`, `lado_pos` | Casilla por posición (mayoría de frames) |
| `conservar` | True si el track tiene identidad y entra a la limpieza |
| `metodo` | `etiqueta` o `posicion` (D17) |
| `motivo` | Razón del descarte: `track genérico sin identidad`, `sin casilla libre compatible`, `punto sin traza`, `video duplicado del punto N` |
| `codigo`, `nombre`, `equipo`, `fondo`, `lado` | Jugador asignado y su casilla según la traza |
| `concuerda_posicion` | 1 si la casilla por posición coincide con la de la traza (control de calidad) |

### `puntos.csv`

| Columna | Descripción |
|---|---|
| `partido`, `punto`, `cancha` | Identificación del punto |
| `punto_equipo_1` | Variable objetivo: 1 si ganó el equipo 1 (D7) |
| `marcador` | Marcador acumulado tras el punto |
| `codigo_j{1,2}_e{1,2}` | Códigos de los cuatro jugadores según la traza |
| `en_traza`, `tiene_video` | El punto existe en la traza o en el video |
| `filas`, `frames`, `tracks`, `tracks_genericos` | Tamaño del video del punto |
| `fps` | 1 / mediana de `duration` |
| `pct_team_faltante`, `pct_pose`, `pct_y_copia_x`, `pct_fuera_cancha` | Porcentajes de calidad por punto |
| `schema_v2` | El punto viene con `schema_version` = 2 |
| `duplicado_de_punto` | Número del punto del que es copia exacta (D16) |
| `y_invalida`, `sin_pose` | Banderas de calidad (sección 1.2): `y` = `x` en más del 95 % de las filas; 0 % de frames con pose |
| `identificados_e1`, `identificados_e2` | Jugadores con track asignado por equipo (0–2) |
| `estado_video` | `disponible`, `sin video`, `video sin traza` o `duplicado` |
| `equipos_completos` | Los 4 jugadores están identificados |
| `un_solo_fondo` | Video disponible pero un equipo sin ningún jugador identificado |

### `video_limpio.parquet`

Una fila = un jugador conservado en un frame (sección 1.2). Solo posición y velocidad (D15); los huecos de ≤5 frames están interpolados y los más largos quedan NaN.

| Columna | Tipo / unidad | Descripción |
|---|---|---|
| `partido`, `punto`, `frame_idx`, `player_id` | — | Llave del frame y del track |
| `codigo`, `equipo`, `fondo`, `lado` | — | Jugador y su casilla según la traza |
| `fps` | frames/s | 1 / mediana de `duration` del track |
| `x_m`, `y_m` | m | Posición suavizada (media móvil 0,5 s) y dentro de la cancha (margen 0,5 m). `y_m` es NaN si la `y` del punto es inválida (D3) |
| `dist_red_m` | m | Distancia a la red (`|y_m − 10|`) |
| `velocidad_mps`, `aceleracion_mps2` | m/s, m/s² | Calculadas sobre la posición suavizada, con límites de 8 m/s y 10 m/s² (D20) |
| `posicion_corregida` | bool | El frame estaba fuera de la cancha o era un salto del tracking |
| `cinematica_corregida` | bool | La velocidad o la aceleración del frame superaban el límite |

### `distribuciones_jugador.csv`

Una fila por jugador con lo que la fase 4 necesita para sortear sus dos variables (3.3, D42–D44).

| Columna | Unidad | Descripción |
|---|---|---|
| `codigo`, `nombre` | — | Código de 6 dígitos y nombre canónico del jugador |
| `puntos` | conteo | Filas jugador × punto con variables de video (D41) |
| `origen` | texto | `jugador` (sus propios parámetros) o `global` (menos de 10 puntos, D42) |
| `p_atacante` | proporción | p de la Bernoulli de `atacante` |
| `metodo_vel` | texto | `parametrica` (su gamma), `bootstrap` (KS rechazó su gamma: se sortea de sus puntos) o `global` |
| `familia_vel`, `scipy_vel` | texto | Familia de `vel_media_mps` y su nombre en `scipy.stats` (vacío si es bootstrap) |
| `parametros_vel` | JSON | Parámetros de `scipy.stats` en orden (forma, desplazamiento, escala); `dis.distribucion_vel(fila)` reconstruye la distribución |
| `p_KS_vel` | p-valor | KS de su distribución contra sus datos (vacío si usa la global) |
| `p90_obs`, `p90_IC95`, `p90_dist` | m/s | Percentil 90 de sus datos, su intervalo del 95 % por bootstrap de sus puntos y el percentil 90 de su distribución |
| `p90_en_IC` | 0/1 | 1 si el percentil 90 de su distribución cae en ese intervalo |

### `processed/tabla_modelo.csv`

Tabla para la fase 2 (1.5, D27): dos filas por punto, una por equipo. Tiene 224 puntos de 24 partidos (448 filas y 36 columnas): los que tienen a los 4 jugadores identificados (D25) y con todas sus variables (D26). No tiene vacíos.

| Columna | Descripción |
|---|---|
| `partido`, `punto` | Identificación del punto |
| `equipo` | Equipo de la fila (1 o 2) |
| `gano` | Objetivo: 1 si ganó el equipo de la fila |
| `propio_<variable>` | Promedio de la variable en los dos jugadores del equipo de la fila |
| `rival_<variable>` | Lo mismo para el equipo rival |

`<variable>` es cualquiera de las 12 que quedan tras la 1.4 (las 14 de `participaciones` menos `dist_red_m` y `pct_quieto`); con el one-hot son 16 columnas por equipo (D21). En las 0/1 el promedio es la proporción del equipo: `propio_atacante` = 0,5 quiere decir que uno de los dos es atacante.

### `processed/ranking_parejas.csv`

Una fila por pareja posible, ordenadas por puntaje (4.3–4.5, D48, D49).

| Columna | Unidad | Descripción |
|---|---|---|
| `pareja` | texto | Nombres de los dos jugadores |
| `nivel_medio` | 0–2 | Promedio del nivel de los dos (0 = primera vez, 1 = principiante, 2 = intermedio o avanzado) |
| `estatura_media` | cm | Promedio de la estatura de los dos |
| `atacante_medio` | proporción | Promedio de la p de atacante de los dos (fase 3) |
| `diferencia_atacante` | proporción | Diferencia absoluta entre las p de atacante de los dos (0 = mismo estilo) |
| `vel_media` | m/s | Promedio de la media de las distribuciones de `vel_media_mps` de los dos |
| `mujeres` | conteo | Cuántas mujeres tiene la pareja (0, 1 o 2); no es variable del modelo |
| `puntaje` | probabilidad | Probabilidad media de ganar el partido contra las 378 parejas sin jugadores en común (promedio de las réplicas de la 4.3) |
| `semiancho_sim` | probabilidad | Semiancho del intervalo t del 95 % de las réplicas: error de la simulación |
| `puntaje_IC95`, `puesto_IC95` | texto | Intervalo del 95 % del puntaje y del puesto en las 100 réplicas de remuestreo de partidos (D49) |
| `pct_primera`, `pct_top10` | % | Réplicas de remuestreo en que la pareja queda primera y entre las 10 mejores |
