# Diccionario de datos

Describe cada columna que usa el proyecto: las tres fuentes crudas que llegan a `data/raw/` (parte A) y las tablas derivadas que genera el notebook [Proyecto_Parejas_Padel.ipynb](../notebooks/Proyecto_Parejas_Padel.ipynb) en `data/interim/` (parte B). Las reglas y decisiones `D#` están en [CLAUDE.md](../CLAUDE.md). La información del CSV de video sale de la documentación y el código del pipeline de visión del profesor (repositorio `padel-vision-analytics-light`), que no se incluye aquí.

- **Parte A — Fuentes crudas:** [A1. Video](#a1-csv-de-video-base_videos_final_202602csv) · [A2. Traza](#a2-traza-de-partidos-traza_partidos_202602xlsx) · [A3. Formulario](#a3-formulario-de-jugadores-datos_jugadores_padel_202602xlsx)
- **Parte B — Tablas derivadas:** `equivalencias_nombres`, `jugadores`, `participaciones`, `identidades_video` y `puntos`.

---

## Parte A — Fuentes crudas (`data/raw/`)

**Leyenda de uso:** **Usar** = entra a la agregación · **Derivar** = solo para calcular otra variable · **Recalcular** = no usar el valor original · **Descartar** = no se usa · **Fuera del plan** = existe pero el plan no la contempla.

### A1. CSV de video `Base_Videos_Final_202602.csv`

Una fila = un jugador en un frame. Un solo CSV consolidado con todos los puntos (718.936 filas, 27 partidos, 272 puntos), con 74 columnas: las 67 del CSV por punto (`Partido_X_Punto_Y_CON_NOMBRES.csv`) más las 7 de la tabla «Columnas agregadas en la base consolidada».

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
| `distance_player_to_net_m` | Distancia a la red | m | Usar |
| `distance_player_to_teammate_m` | Distancia al compañero | m | Usar (recalcular faltantes desde las posiciones) |
| `player_displacement` | Desplazamiento entre frames | m | Recalcular sobre la trayectoria suavizada |
| `player_speed_mps`, `player_acceleration_mps2` | Velocidad y aceleración | m/s, m/s² | Recalcular (ruido por cuantización) |
| `prev_x`, `prev_y`, `prev_speed` | Valores del frame anterior | — | Descartar |
| `ball_position_x`, `ball_position_y`, `ball_position_x_prev`, `ball_position_y_prev` | Posición de la pelota | m | Descartar |
| `ball_displacement`, `ball_speed_mps`, `distance_ball_to_net`, `distance_player_to_ball_m` | Movimiento y distancias de la pelota | m, m/s | Descartar |
| `player_hits_ball`, `time_since_last_hit` | Golpe detectado y tiempo desde el último | bool, s | Descartar (golpes por picos de `racket_speed`) |
| `head_x/y`, `shoulder_x/y`, `elbow_x/y`, `wrist_x/y`, `racket_tip_x/y` | Keypoints H, S, E, W, R | px | Derivar |
| `upper_arm_length`, `forearm_length`, `arm_extension` | dist(S,E), dist(E,W), dist(S,W) | px | Descartar |
| `racket_length_visible` | dist(W,R) | px | Fuera del plan |
| `elbow_angle` | Ángulo (S, E, W) | ° | Usar |
| `shoulder_angle` | Ángulo (E, S, H) | ° | Usar |
| `racket_vector_x`, `racket_vector_y` | (R − W) unitario | — | Descartar |
| `racket_orientation_deg` | atan2(Ry − Wy, Rx − Wx) | ° | Usar |
| `racket_direction` | Orientación en 8 direcciones | categórica | Usar (proporción por dirección) |
| `racket_height_relative_to_head` | Hy − Ry (> 0 = sobre la cabeza) | px | Derivar (usar `racket_height_norm`) |
| `arm_extension_ratio` | dist(S,W) / (dist(S,E) + dist(E,W)) · 100 | % | Fuera del plan |
| `forearm_orientation_deg`, `upper_arm_orientation_deg` | Orientación de antebrazo y brazo | ° | Descartar |
| `wrist_above_head` | Wy < Hy | bool | Usar |
| `racket_above_shoulder` | Ry < Sy | bool | Usar (% de juego aéreo) |
| `wrist_height_rel_shoulder` | Sy − Wy | px | Derivar (usar `wrist_height_norm`) |
| `wrist_height_norm` | `wrist_height_rel_shoulder` / `body_scale` | — | Usar |
| `racket_height_norm` | `racket_height_relative_to_head` / `body_scale` | — | Usar |
| `body_scale` | dist(H,S) | px | Derivar (normalizador de todo lo que está en px) |
| `pose_confidence` | Confianza media de los keypoints | 0–1 | Usar (calidad) |
| `pose_available` | Hubo pose en el frame | bool | Derivar (`pose_coverage`) |
| `racket_displacement` | dist(R_t, R_t−1) | px | Derivar |
| `racket_speed` | `racket_displacement` / `duration` | px/s | Usar (÷ `body_scale`, p90/p95, picos) |
| `wrist_speed` | dist(W_t, W_t−1) / `duration` | px/s | Usar (÷ `body_scale`, p90/p95) |
| `racket_acceleration`, `elbow_speed` | Δ`racket_speed` / `duration`, dist(E_t, E_t−1) / `duration` | px/s², px/s | Fuera del plan |
| `elbow_angular_velocity` | Δ`elbow_angle` / `duration` | °/s | Usar (p90/p95) |
| `racket_angular_velocity` | Δ`racket_orientation_deg` (ajustado a ±180) / `duration` | °/s | Usar (p90/p95) |

#### Columnas agregadas en la base consolidada

| Columna | Significado | Uso |
|---|---|---|
| `partido`, `punto` | Partido y punto del video; se cruzan con `ID_PARTIDO`, `ID_PUNTO` de la traza | Usar (llave) |
| `nombre_norm` | `player_name` sin tildes y en minúsculas. Hay etiquetas genéricas `player_N` y `jugador N` | Derivar (identidad, ver D12) |
| `schema_version` | Versión del esquema; solo vale 2 en los partidos 10, 13 y 39 (10 % de las filas) | Descartar |
| `player_speed_implausible`, `ball_speed_implausible`, `row_implausible` | Controles de plausibilidad; solo existen donde `schema_version` = 2 | Descartar |

#### Particularidades por partido

- **5:** el punto 10 es una copia exacta del punto 9 (D7).
- **8 y 34:** solo se detecta la pareja BOTTOM y `team` está vacío. Las etiquetas de nombre no corresponden a la traza (D12).
- **17:** `player_position_y` es una copia de `player_position_x`, no hay pose y las etiquetas son `Jugador 1–4`.
- **27:** el equipo top aparece en `y` ≈ 15 (homografía desplazada); `team` sí es correcto.
- **32:** 300 frames por punto con fps de 11,8 a 39,9.

La detección automática de estos casos está en `src/video.py` (`diagnosticar_puntos`).

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

Una fila = un jugador (29 filas), en la hoja `Respuestas de formulario 1`. La hoja `Hoja 1` es un agregado de resultados que no se usa. Las columnas que se usan, las que se descartan y las reglas de imputación están en [CLAUDE.md](../CLAUDE.md) (secciones «Reglas del formulario y la traza» y «Registro de decisiones»). Las escalas ordinales están en `src/fuentes.py` (`ORDINALES`).

---

## Parte B — Tablas derivadas (`data/interim/`)

CSV UTF-8 con BOM. Ninguna se versiona; se regeneran ejecutando el notebook.

| Tabla | Grano | Filas | Sección |
|---|---|---|---|
| `equivalencias_nombres.csv` | fuente × nombre original | 89 | 1.3 |
| `jugadores.csv` | jugador | 32 | 1.5 |
| `participaciones.csv` | jugador × punto (de la traza) | 1.132 | 1.4 / 1.7 |
| `identidades_video.csv` | track de video `(partido, punto, player_id)` | 1.201 | 1.7 |
| `puntos.csv` | punto (unión externa traza–video) | 287 | 1.7 |

### `equivalencias_nombres.csv`

| Columna | Descripción |
|---|---|
| `fuente` | `formulario`, `traza` o `video` |
| `nombre_original` | Nombre tal como aparece en la fuente |
| `nombre` | Nombre canónico (completo, D9) |
| `tipo_cruce` | `exacto` (igual tras normalizar), `alias` (variante registrada) o `sin cruzar` (buscar en Teams) |
| `codigo` | Código de estudiante de 6 dígitos (texto) |

### `jugadores.csv`

| Columna | Tipo | Descripción |
|---|---|---|
| `codigo` | texto | Código de estudiante, 6 dígitos (D8). Llave |
| `nombre` | texto | Nombre canónico |
| `genero` | categórica | `Femenino` / `Masculino` |
| `mujer` | 0/1 | 1 si `genero` = Femenino |
| `edad` | entero | Años |
| `estatura_cm` | entero | Estatura en cm |
| `peso_kg` | real | `TALLA` del formulario = peso en kg (D1) |
| `imc` | real | peso / (estatura en m)² |
| `mano_dominante` | categórica | `Derecha` / `Izquierda` |
| `zurdo` | 0/1 | 1 si es zurdo (solo 2 jugadores) |
| `nivel_padel` | ordinal 0–3 | Primera vez, Principiante, Intermedio, Avanzado |
| `tiempo_jugando_padel` | ordinal 0–2 | <6 meses, 6 meses–1 año, >1 año |
| `experiencia_padel` | 0/1 | Tenía experiencia previa en pádel |
| `otro_deporte_raqueta` | categórica | `No`, `Tenis`, `Tenis de mesa`, `Squash`, `Otro` |
| `practica_otra_raqueta` | 0/1 | 1 si practica otro deporte de raqueta |
| `estado_fisico` | ordinal 0–3 | Malo, Regular, Bueno, Excelente |
| `frecuencia_deporte` | ordinal 0–3 | 0 veces o no practica (D10), 1–2, 3–4, >4 veces por semana |
| `lesiones_ly` | categórica | Lesiones en el último año (`Sí` / `No`) |
| `lesion_ultimo_anio` | 0/1 | 1 si `lesiones_ly` = Sí |
| `imputado` | bool | True para los 3 jugadores sin formulario (D4) |

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
| `metodo_identidad` | `etiqueta` o `posicion` (D12); vacío si no hay track |

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
| `metodo` | `etiqueta` o `posicion` (D12) |
| `motivo` | Razón del descarte: `track genérico sin identidad`, `sin casilla libre compatible`, `punto sin traza`, `video duplicado del punto N` |
| `codigo`, `nombre`, `equipo`, `fondo`, `lado` | Jugador asignado y su casilla según la traza |
| `concuerda_posicion` | 1 si la casilla por posición coincide con la de la traza (control de calidad) |

### `puntos.csv`

| Columna | Descripción |
|---|---|
| `partido`, `punto`, `cancha` | Identificación del punto |
| `punto_equipo_1` | Variable objetivo: 1 si ganó el equipo 1 (D3) |
| `marcador` | Marcador acumulado tras el punto |
| `codigo_j{1,2}_e{1,2}` | Códigos de los cuatro jugadores según la traza |
| `en_traza`, `tiene_video` | El punto existe en la traza o en el video |
| `filas`, `frames`, `tracks`, `tracks_genericos` | Tamaño del video del punto |
| `fps` | 1 / mediana de `duration` |
| `pct_team_faltante`, `pct_pose`, `pct_y_copia_x`, `pct_fuera_cancha` | Porcentajes de calidad por punto |
| `schema_v2` | El punto viene con `schema_version` = 2 |
| `y_mediana_top` | Mediana de `y` del equipo `top` (debe ser < 10) |
| `duplicado_de_punto` | Número del punto del que es copia exacta (D7) |
| `y_invalida`, `sin_pose`, `y_top_distorsionada`, `fps_no_estandar` | Banderas de calidad (sección 1.6) |
| `identificados_e1`, `identificados_e2` | Jugadores con track asignado por equipo (0–2) |
| `estado_video` | `disponible`, `sin video`, `video sin traza` o `duplicado` |
| `equipos_completos` | Los 4 jugadores están identificados |
| `un_solo_fondo` | Video disponible pero un equipo sin ningún jugador identificado |
