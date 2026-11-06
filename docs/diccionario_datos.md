# Diccionario de datos de entrada

Describe las fuentes que llegan a `data/raw/` y cómo se usa cada columna según las reglas de [CLAUDE.md](../CLAUDE.md).
La información del CSV de video sale de la documentación y el código del pipeline de visión del profesor (repositorio `padel-vision-analytics-light`), que no se incluye aquí.

**Leyenda de uso:** **Usar** = entra a la agregación · **Derivar** = solo para calcular otra variable · **Recalcular** = no usar el valor original · **Descartar** = no se usa · **Fuera del plan** = existe pero el plan no la contempla.

## 1. CSV de video `Partido_X_Punto_Y_CON_NOMBRES.csv`

Una fila = un jugador en un frame. 67 columnas.

### Cómo se generan (importa para la limpieza)

- **Sistema de coordenadas (m):** homografía de la imagen a la cancha. `x` ∈ [0, 10] a lo ancho, `y` ∈ [0, 20] a lo largo, (0, 0) = esquina del fondo lejano izquierdo. Red en `y = 10`. El equipo `top` está en `y < 10`.
- **fps:** se leen del video. `duration` = 1/fps; no es constante entre videos.
- **`team`:** se asigna en cada frame ordenando a los jugadores por `y` (los 2 de menor `y` = `top`). Si en el frame no hay exactamente 4 jugadores detectados, queda vacío. Eso explica los frames sin `team`.
- **`distance_player_to_teammate_m`:** se calcula con la misma agrupación por `y`; queda vacía cuando no se puede formar la pareja.
- **`distance_player_to_net_m`:** `|y − 10|`.
- **`zone`:** `back_court` si el jugador está a más de 6,95 m de la red; si no, `left_box`/`right_box` según el lado, con la referencia invertida entre `top` y `bottom`. Es geométrica, no un rol.
- **Pelota:** sus posiciones se interpolan linealmente en todos los huecos y se rellenan el inicio y el final del punto con la primera y la última detección. Por eso parecen relleno.
- **Velocidades:** son diferencias entre frames consecutivos divididas por `duration` (sin suavizado).
- **Pose:** un modelo YOLO11s-pose detecta solo el brazo de juego (hombro, codo, muñeca, punta de raqueta y cabeza; ver [assets/pose_keypoints_map.jpg](assets/pose_keypoints_map.jpg)) y se asocia al jugador por IoU ≥ 0,3. Los keypoints con confianza 0 quedan vacíos. Todo lo de pose está en píxeles.

### Columnas

Notación de pose: S = hombro, E = codo, W = muñeca, R = punta de raqueta, H = cabeza. Las coordenadas `y` de imagen crecen hacia abajo.

| Columna | Significado | Unidad | Uso |
|---|---|---|---|
| `frame_idx` | Número de frame | — | Usar (orden temporal) |
| `duration` | Duración de un frame | s | Derivar (fps = 1/`duration`) |
| `total_frames`, `video_duration` | Totales del video | —, s | Descartar |
| `player_id` | ID estable del jugador dentro del punto (1–4) | — | Usar (llave dentro del punto) |
| `player_name` | Nombre del jugador | — | Derivar (solo para cruzar a `CODIGO_ESTUDIANTE`) |
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

## 2. Traza de partidos (`CANCHA1.csv`, `CANCHA2.csv`)

Una fila = un punto. Según el script del profesor ([referencia/normalize_traza_profesor.py](../referencia/normalize_traza_profesor.py)), se lee con `encoding="latin-1"` y `sep=";"`.

| Columna | Significado |
|---|---|
| `ID_PARTIDO` | Partido (pares en la cancha 2, impares en la 1) |
| `ID_PUNTO` | Número de punto dentro del partido |
| `ID_JUGADOR_{1,2}_EQUIPO_{1,2}` | **Nombre** del jugador (no es un código; se cruza con la tabla de equivalencias) |
| `POSICION_JUGADOR_{1,2}_EQUIPO_{1,2}` | `LEFT` / `RIGHT` |
| `EQUIPO_{1,2}_POSICION` | `TOP` / `BOTTOM` |
| `PUNTO_EQUIPO_1`, `PUNTO_EQUIPO_2` | 1 si el equipo ganó el punto. `PUNTO_EQUIPO_1` es la variable objetivo |

El script convierte la traza a formato largo jugador × punto (`punto_ganado`), pero asigna los IDs por nombre. Sirve como referencia del formato; en este proyecto la llave es `CODIGO_ESTUDIANTE`.

## 3. Formulario de jugadores

Una fila = un jugador (29 filas). Las columnas que se usan, las que se descartan y las reglas de imputación están en [CLAUDE.md](../CLAUDE.md) (secciones "Reglas del formulario y la traza" y "Registro de decisiones").
