# Proyecto final de analítica: parejas ideales de pádel

Encontrar las parejas que maximizan la probabilidad de ganar el torneo según su estilo de juego. Se parte de tres fuentes: datos de video (pose y posición por frame), la traza de partidos y el formulario de jugadores.

**¿Por dónde empezar?** El trabajo completo (código, resultados y reporte, por fases) está en un solo notebook: [notebooks/Proyecto_Parejas_Padel.ipynb](notebooks/Proyecto_Parejas_Padel.ipynb). Este README explica los archivos del repositorio y concentra las reglas de datos, el [registro de decisiones](#registro-de-decisiones) y los [pendientes](#supuestos-y-pendientes).

## Contenido

1. [Fases](#fases)
2. [Archivos del repositorio](#archivos-del-repositorio)
3. [Datos e instalación](#datos-e-instalación)
4. [Reglas de datos](#reglas-de-datos)
5. [Registro de decisiones](#registro-de-decisiones)
6. [Supuestos y pendientes](#supuestos-y-pendientes)

## Fases

| Fase | Contenido | Estado |
|---|---|---|
| 1. Manejo de datos | Fuentes, entendimiento, identidad, traza, formulario, calidad e identidad del video, limpieza, variables de jugador y de pareja, variables para el modelo (1.1–1.11) | Hecho |
| 2. Predictiva | Modelo base contra modelos avanzados, ensamblaje y bootstrap sobre el ganador del punto | Hecho (2.1–2.8). En test, ningún modelo supera al naive (ver pendientes) |
| 3. Distribuciones | Distribución empírica por jugador y por variable aleatoria | Hecho (3.1–3.6). Quedan 3 decisiones para la fase 4 (ver pendientes) |
| 4. Simulación | Combinaciones de parejas evaluadas con el modelo de la fase 2 | Pendiente |

## Archivos del repositorio

```
README.md                         Este archivo: archivos, reglas de datos, decisiones y pendientes
CLAUDE.md                         Reglas de trabajo para el asistente de IA (Claude Code) e hitos
requirements.txt                  Librerías de Python necesarias
.gitignore                        Lo que no se sube a git
notebooks/
  Proyecto_Parejas_Padel.ipynb    Notebook único de trabajo: todas las fases y el reporte
  referencias/                    Notebooks del profesor que se replican en la fase 2 (solo consulta)
src/                              Funciones del pipeline que usa el notebook
  __init__.py                     Marca src/ como paquete y lista sus módulos
  datos.py                        Rutas, constantes de cancha, lectura de fuentes, utilidades
  identidades.py                  Nombres canónicos, alias y códigos de estudiante
  fuentes.py                      Limpieza de la traza y del formulario
  video.py                        Calidad, identidad, pelota, atípicos y limpieza del video
  variables.py                    Variables de estilo: jugador × punto, pareja × punto y filtros
  modelo.py                       Fase 2: evaluación común (validación por partido, métricas y gráficos)
  distribuciones.py               Fase 3: exploración, ajuste de distribuciones, por jugador y bootstrap
docs/
  diccionario.md                  Diccionario de todas las columnas (fuentes y tablas derivadas)
  pose_keypoints_map.jpg          Imagen de los puntos de pose que usa el diccionario
data/
  raw/                            Datos de entrada (versionados, D19)
    Traza_Partidos_202602.xlsx
    Datos_Jugadores_Padel_202602.xlsx
    video/partido_XX.parquet      Video en Parquet, un archivo por partido (27)
  interim/                        Tablas que genera el notebook (no se suben)
  processed/                      Tabla final para el modelo (no se sube)
```

### Qué es y qué hace cada archivo

| Archivo | Qué es | Qué hace / cuándo usarlo |
|---|---|---|
| `notebooks/Proyecto_Parejas_Padel.ipynb` | El documento de trabajo del proyecto. | Se ejecuta de principio a fin (~20 min; la fase 1 tarda ~3 min, la fase 3 menos de 1 min y el resto son las búsquedas de hiperparámetros de la 2.5 y la 2.6 y el ensamblaje de la 2.7). Cada sección tiene Objetivo → Decisiones → Código → Resultado. La sección 1.2 reúne el entendimiento de los datos (incluye la evidencia para descartar la pelota y el diagnóstico de atípicos); la 1.8 es la limpieza del video. Los anexos A y B enlazan a este README; el anexo C lista las tablas generadas. |
| `notebooks/referencias/` | Notebooks de clase del profesor. | Solo consulta: la fase 2 replica su estructura y su código. No se ejecutan ni forman parte del pipeline. |
| `CLAUDE.md` | Instrucciones para el asistente de IA. | Cómo debe trabajar (preguntar antes de decidir, verificar, mantener el repositorio simple) y los hitos del proyecto. No tiene decisiones concretas: remite a este README. |
| `src/datos.py` | La base del pipeline. | Rutas de `data/`, semilla (2026), medidas de la cancha, lectura de las fuentes (`leer_video`, `leer_traza`, `leer_formulario`, `iterar_video`), conversión del CSV de video a Parquet (`convertir_video_a_parquet`) y utilidades de reporte y escritura (`resumen_tabla`, `comparar`, `perfil_columnas`, `guardar_tabla`). |
| `src/identidades.py` | La tabla de equivalencias de jugadores. | Lleva cualquier nombre de cualquier fuente a su nombre canónico y a su código de 6 dígitos. Aquí se agregan alias nuevos, jugadores sin formulario y correcciones de código. |
| `src/fuentes.py` | La limpieza de las dos fuentes pequeñas. | Traza: controles de consistencia, erratas, códigos y formato largo jugador × punto. Formulario: variables usables, escalas ordinales, imputación de los jugadores sin formulario e IMC. |
| `src/video.py` | El tratamiento del video. | Calidad por punto, identidad de cada track según la traza, evidencia de la pelota, diagnóstico de atípicos y limpieza frame × jugador (suavizado, límites físicos, pose, variables combinadas, golpes). |
| `src/variables.py` | Las variables de estilo de juego. | Calcula las variables jugador × punto (estilo, movimiento, pose, golpes y ritmo; sección 1.9), arma la tabla pareja × punto relativa al rival y aplica los filtros sin mirar el resultado (sección 1.10). |
| `src/modelo.py` | La evaluación de la fase 2. | `predecir_cv` (entrena con un partido fuera e imputa con la mediana del train), `metricas` (AUC por pliegue, log-loss, Brier, accuracy, F1), `graficar_evaluacion` (AUC por partido o ROC, calibración y matriz de confusión) ; para la 2.7, `ensamblar_cv` (promedio de probabilidades con pesos iguales, ∝ 1 − log-loss u óptimos), `pesos_log_loss` y `pesos_optimos`; para la 2.8, la clase `Ensamblaje` (modelo final y el de la fase 4); y `silenciar_aviso_paralelo`, que quita de la salida un aviso de scikit-learn 1.9 + joblib que se repetía miles de veces en la 2.5 (no cambia resultados). Todos los modelos de la 2.3 a la 2.8 pasan por aquí. |
| `src/distribuciones.py` | La fase 3. | `observaciones` (jugador × punto con video), exploración univariada, bivariada y multivariada con sus gráficos, `ajustar_todo` (MLE de las candidatas, KS o chi², AIC y elección, D52–D56), `ajustar_por_jugador` (D57), `bootstrap_vs_parametrica`, `tabla_final` y `distribucion` (reconstruye una distribución guardada, para la fase 4). |
| `src/__init__.py` | Archivo técnico de Python. | Permite hacer `from src import ...` desde el notebook. |
| `docs/diccionario.md` | El diccionario de datos. | Parte A: cada columna de las fuentes crudas (unidad, cómo se genera y uso). Parte B: cada columna de las tablas que genera el notebook. |
| `docs/pose_keypoints_map.jpg` | Imagen de referencia. | Puntos del cuerpo que detecta el modelo de pose (hombro, codo, muñeca, raqueta y cabeza). |
| `requirements.txt` | Lista de librerías. | `pip install -r requirements.txt`. |
| `.gitignore` | Configuración de git. | Sube los Excel y el Parquet del video; deja fuera el CSV original, las tablas generadas, los cachés y los entornos. |

### Tablas que genera el notebook (`data/interim/`)

| Tabla | Una fila por | Contenido |
|---|---|---|
| `equivalencias_nombres.csv` | nombre de cada fuente | Nombre original → nombre canónico → código |
| `jugadores.csv` | jugador (32) | Variables fijas del formulario, codificadas e imputadas, más el IMC y su estilo atacante/defensivo |
| `participaciones.csv` | jugador × punto (1.132) | Equipo, fondo, lado, compañero, si ganó el punto, su track de video, la calidad de su limpieza y sus variables de estilo (1.9) |
| `identidades_video.csv` | track de video (1.201) | A qué jugador corresponde cada track, cómo se asignó y si se conserva |
| `puntos.csv` | punto (287) | Ganador, jugadores, estado del video y banderas de calidad |
| `video_limpio.parquet` | frame × jugador (705.888) | Video limpio: posiciones suavizadas, velocidad, pose, variables combinadas y golpes |
| `parejas_punto.csv` | punto completo (244) | Todas las variables de cada pareja (`e1_`, `e2_`) y su diferencia con el rival (`dif_`) |
| `filtros_variables.csv` | variable descartada (83) | Qué variable salió en los filtros de la 1.10 y por qué |
| `processed/tabla_modelo.csv` | punto completo (244) | Tabla para el modelo: objetivo y las 161 variables que pasan los filtros |
| `distribuciones_global.csv` | componente (11) | Fase 3: distribución elegida con todos los jugadores, parámetros, prueba, decisión y resumen por jugador |
| `distribuciones_jugador.csv` | jugador × componente (300) | Fase 3: método (paramétrica, bootstrap o global), familia y parámetros de cada jugador, para la fase 4 |

El detalle de cada columna está en la parte B de [docs/diccionario.md](docs/diccionario.md).

## Datos e instalación

Los datos están versionados en este repositorio, que es privado (D19). Basta con clonarlo o hacer `git pull`.

- `data/raw/Traza_Partidos_202602.xlsx` y `data/raw/Datos_Jugadores_Padel_202602.xlsx`: tal como llegaron.
- `data/raw/video/partido_XX.parquet`: el CSV de video `Base_Videos_Final_202602.csv` (655 MB, demasiado grande para GitHub) convertido sin cambios, un archivo por partido (172 MB en total).
- **Si llega un CSV de video nuevo:** se copia en `data/raw/` con ese nombre, se ejecuta `datos.convertir_video_a_parquet()` y se suben los Parquet.

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
source .venv/bin/activate     # macOS / Linux
pip install -r requirements.txt
```

Luego se ejecuta el notebook de principio a fin; las tablas derivadas quedan en `data/interim/`.

## Reglas de datos

Reglas concretas acordadas. Cada una indica su decisión `D#` ([registro](#registro-de-decisiones)).

### Fuentes y cobertura

| Fuente | Granularidad | Contenido |
|---|---|---|
| Video | frame × jugador | 718.936 filas y 74 columnas; 27 partidos y 272 puntos. El fps varía por video (`duration` = 1/fps). |
| Traza | punto | Hojas `CANCHA_1` (16 partidos, 164 puntos) y `CANCHA_2` (11 partidos, 119 puntos); partidos a 7 puntos. La hoja `Original` programa 39 partidos, de los cuales 12 no tienen traza ni video. `TDs` y `Resultados` no se usan. |
| Formulario | jugador | 29 filas; la hoja `Hoja 1` no se usa. |

- **Llave de unión:** `(partido, punto, codigo)`, con `codigo` = `CODIGO_ESTUDIANTE` como texto de 6 dígitos. Ningún nombre es llave: todo nombre pasa por `src/identidades.py`.
- **Coordenadas del video:** metros, con `x` ∈ [0, 10], `y` ∈ [0, 20] y red en `y = 10`. BOTTOM está en `y > 10` y, para ambos equipos, RIGHT en `x > 5` (D13).
- **Cobertura:** de los 283 puntos de la traza:
  - 16 no tienen video utilizable: 15 sin video y el 5-10, duplicado.
  - 23 tienen video incompleto: 16 solo con la pareja BOTTOM (partidos 8 y 34) y 7 con 3 de los 4 jugadores.
  - 244 tienen a los 4 jugadores identificados; solo esos entran al modelo pareja contra pareja (D20).

### Casos especiales del video

No hay tratamientos por partido (D26, D30). Los casos especiales se detectan con reglas generales que el notebook calcula y muestra:
- **Duplicados exactos** de un punto dentro del partido: se conserva el de menor número (D7). Hoy aplica al partido 5, punto 10.
- **`y` igual a `x`** en más del 95 % de las filas de un punto: la `y` se marca faltante. Hoy aplica al partido 17.
- **Etiquetas de nombre que no concuerdan con la traza:** la identidad se asigna por posición (D12). Hoy aplica a los partidos 8 (etiquetas de jugadores que no están en su alineación), 17 (`Jugador 1–4`) y 34 (concordancia del 46 %; el track etiquetado «David Neisa» queda en la casilla de Alejandra Deza).
- **Puntos de video sin traza:** se excluyen (D15).

### Identidad de los jugadores

- **Nombres (D9):** las variantes de un nombre son el mismo jugador. El nombre canónico es el completo, con dos apellidos cuando alguna fuente lo tiene (`NOMBRES_CANONICOS`).
- **Códigos (D4, D8):** todos tienen 6 dígitos y se guardan como texto (`CORRECCIONES_CODIGO`, `JUGADORES_SIN_FORMULARIO`).
- **Tracks del video (D12, D14):**
  - Se usa la etiqueta del video si el partido tiene ≥50 % de filas con nombres de la alineación y ≥80 % de concordancia con la casilla (fondo, lado) de la traza.
  - Si no, la identidad se asigna por posición contra la traza.
  - Los `Player_N` y los tracks sin casilla compatible se descartan.
- **Búsqueda en Teams (proceso recurrente):** todo jugador que no cruce (con videos, partidos o formularios nuevos) se lista en la sección 1.3 y se busca en Teams. Con el nombre y el código confirmados, se agrega a `src/identidades.py` y se vuelve a correr la fase 1. Hoy no hay jugadores sin cruzar.

### Limpieza del video (sección 1.8)

- **Solo se limpian los tracks conservados** en `identidades_video.csv`.
- **Pelota (D24):** se descartan todas sus columnas, incluida `player_hits_ball`. Cerca del 61 % de sus posiciones son relleno o interpolación lineal (evidencia en 1.2.7). Los golpes se aproximan con picos de velocidad de raqueta.
- **Fuera de la cancha (D29):** las posiciones hasta 0,5 m afuera se recortan al borde; las más lejanas pasan a NaN.
- **Suavizado (D29):** media móvil centrada de 0,5 s de `x` e `y`. La velocidad y la aceleración se recalculan sobre esa trayectoria, con el fps de cada punto. No se usan las columnas originales, que son ruido de cuantización.
- **Atípicos (D23):**
  - Límites: velocidad 8 m/s, aceleración 10 m/s² y velocidades angulares 3.000 °/s.
  - Velocidades en píxeles ÷ `body_scale`: percentil 99,5 de cada jugador.
  - Tratamiento: el valor pasa a NaN y se interpola si el hueco es de ≤5 frames. Primero se quitan las posiciones con saltos del tracking y luego se recalcula todo.
- **Píxeles:** toda medida en píxeles se divide por `body_scale`. Los ángulos y las variables `*_norm` ya son comparables.
- **Pose:** se interpolan solo huecos de ≤5 frames; los indicadores booleanos no se interpolan. `pose_coverage` se guarda por jugador y punto.
- **Variables combinadas (D25):**
  - `velocidad_mps`, `aceleracion_mps2` y `desplazamiento_m` (desde `x` e `y`);
  - `indice_vel_golpe` (promedio de las velocidades de raqueta, muñeca y codo normalizadas);
  - `juego_aereo` (raqueta sobre el hombro o muñeca sobre la cabeza).
- **Rol (D29):** el lado de la traza: drive = RIGHT y revés = LEFT. `en_su_lado` indica si en cada frame está en su lado.
- **Golpes (D29):** máximos locales de la velocidad de raqueta normalizada por encima del p90 del jugador en el punto, separados al menos 0,4 s.
- **Faltantes menores:** `team` sale del fondo de la traza y la distancia al compañero se recalcula desde las posiciones suavizadas.
- **Descartadas:** `duration` (solo para el fps), `total_frames`, `video_duration`, `prev_*`, las coordenadas en píxeles, `upper_arm_length`, `forearm_length`, `arm_extension`, `racket_vector_x/y`, las orientaciones de antebrazo y brazo, `zone` (no es un rol), `schema_version` y `*_implausible`.

### Formulario y traza

- **Fuga de datos:** `Puntos_Total`, `Partidos_Jugados` y `Puntos_Por_Partido` nunca son predictoras; solo validan la traza (`Puntos_Total` coincide en 27 de 27 jugadores).
- **Usables:** `ESTATURA`, `TALLA` (peso, D1), `MANO_DOMINANTE` (solo 2 zurdos), `NIVEL_ACTUAL_PADEL`, `TIEMPO_JUGANDO_PADEL`, `EXPERIENCIA_PADEL`, `PRACTICA_OTRO_DEPORTE_RAQUETA`, `ESTADO_FISICO`, `FRECUENCIA_DEPORTE`, `EDAD`, `GENERO` y `LESIONES_LY`. Las escalas ordinales están en `ORDINALES` (D10, D11).
- **Descartadas:** ciudad, programa, semestre, razón y objetivo de participación, interés y clases de pádel, desayuno, género musical y comentarios. También `RITMO_CARDIACO`: tiene pocos datos (12 de 29) y no interesa (confirmado por el equipo el 2026-11-07).
- **Lado de cancha:** el equipo 1 siempre está en BOTTOM. Entra como variable de control, sin conclusiones fuertes.
- **Traza:** pasa todos los controles (un ganador por punto, marcador igual al acumulado, puntos consecutivos, lados fijos por partido). El equipo 1 gana el 43,1 % de los puntos.

### Variables (secciones 1.9 y 1.10)

Granularidad: **frame × jugador → jugador × punto → pareja × punto**, con la pareja rival como contexto.

- **Estilo atacante/defensivo (D21):** atacante si pasa al menos tanto tiempo delante de la línea de saque (< 6,95 m de la red) como detrás. Se calcula por jugador × punto; el jugador es atacante si lo fue en al menos el 50 % de sus puntos. Las secciones red (0–3 m), transición (3–6,95 m) y fondo (6,95–10 m) quedan como variables descriptivas.
- **Definiciones (D31):**
  - quieto = velocidad por debajo de la mediana global (0,37 m/s); ráfaga = tramo de al menos 0,5 s por encima del p90 global (1,28 m/s);
  - área = cuadros de 0,5 m pisados;
  - variabilidad de ángulos = desviación estándar (circular para la raqueta);
  - dirección de raqueta en 3 grupos;
  - mínimos: track de al menos 2 s y cobertura de pose de al menos el 50 % para las variables de pose.
- **Ritmo de juego (D22):** duración del punto y golpes por segundo, por jugador y por pareja.
- **Jugador × punto:** profundidad media y desviación, % en cada sección, lado, área cubierta, distancia recorrida, % quieto, ráfagas rápidas, velocidad p90, % de juego aéreo, proporción por dirección de raqueta, variabilidad de ángulos, `indice_vel_golpe` p90/p95, golpes y `pose_coverage`.
- **Pareja × punto (D32):**
  - suma y diferencia absoluta de cada variable del jugador, más la razón mín/máx para velocidad, golpes y área;
  - número de atacantes;
  - cohesión (distancia entre compañeros);
  - sincronía (correlación de velocidad y de profundidad);
  - fijas de pareja, incluido `en_lado_mano` (diestro a la derecha o zurdo a la izquierda);
  - duración del punto como contexto.
  - Una fila por punto: `e1_`, `e2_` y `dif_` (equipo 1 − equipo 2).
- **Filtros sin mirar el resultado (D33):** faltantes > 20 %, valor dominante ≥ 95 % y correlación de Pearson |r| ≥ 0,85. Se entregan todas las que pasan; los faltantes se imputan en la fase 2.
- **Variables para las fases 2 a 4 (D34, sección 1.11):** 19 variables `dif_` en `variables.VARIABLES_MODELO`, todas reconstruibles desde cada jugador con `variables.combinar_pareja` (verificado: error 0 en los 244 puntos).
- **Selección final (D27):** Lasso (L1) y Ridge (L2) dentro de la validación de la fase 2, sobre esas 19, para quedar con 10 a 15 variables.

### Modelado (fases 2 a 4)

- **Objetivo:** `PUNTO_EQUIPO_1` por punto (D3). La probabilidad de ganar el partido a 7 puntos se deriva de p(ganar el punto).
- **Partición (D35, sección 2.1):** test = 5 de los 25 partidos (20 %, `GroupShuffleSplit`, `random_state=0`), que no se toca hasta la 2.8. Validación cruzada dentro de los 20 partidos de train, dejando un partido fuera (`LeaveOneGroupOut`, 20 pliegues). Nunca se mezclan puntos del mismo partido.
- **Modelos (D28):** un modelo base (logística regularizada) contra modelos avanzados (Random Forest y otros), un ensamblaje de modelos y bootstrap para medir la fiabilidad.
- **Métricas (D36):** AUC por pliegue, log-loss, Brier, calibración e importancia de variables. La referencia mínima es el naive de la 2.2: log-loss 0,696 y Brier 0,251 en la validación cruzada.
- **Modelo final (D47–D49, secciones 2.7 y 2.8):** `ens_optimo` = 0,745 Gradient Boosting + 0,255 LASSO, elegido en validación antes de mirar el test. En el test (usado una sola vez) queda peor que el naive (log-loss 0,766 contra 0,674) y no es simétrico al intercambiar las parejas. Para la fase 4 se reentrena con los 244 puntos (`modelo_fase4`).
- **Fase 3:** distribución empírica por jugador y por variable, comparando el bootstrap empírico contra una paramétrica ajustada (reglas en la sección siguiente).
- **Fase 4:** ranking de parejas por probabilidad media de ganar contra el resto, con intervalos de incertidumbre.
- Las parejas casi no se repiten (30 jugadores): el modelo aprende de características de pareja, no de identidades.

### Distribuciones (fase 3)

- **Qué se modela:** las 7 variables de jugador × punto con las que se arman las 19 `dif_` (`prof_media_m`, `prof_sd_m`, `pct_en_su_lado`, `vel_p90`, `area_m2`, `pct_quieto`, `rafagas_por_min`) más `atacante`. Los atributos fijos no se sortean.
- **Datos (D50):** todos los jugador × punto con video (1.015; 979 con profundidad y movimiento), no solo los 244 puntos completos. Ajuste global y por jugador; por jugador solo con ≥ 10 observaciones, si no la global. Los códigos se leen como texto (`dis.leer_participaciones`, D8).
- **Orden temporal (D51):** `(partido, punto, equipo, lado)`; media móvil de 30; estacionaria = ADF rechaza (p < 0,05) y KPSS no rechaza (p ≥ 0,05); ACF de rezago 1 dentro de cada jugador.
- **Componentes (D53):** `pct_en_su_lado` = Bernoulli P(= 100) + déficit 100 − x; `rafagas_por_min` = Bernoulli P(= 0) + positivas; `area_m2` = conteo de cuadros extra (área / 0,25 − 1); `atacante` = Bernoulli.
- **Candidatas (D54):** normal, uniforme, triangular, exponencial, gamma, Erlang, chi², lognormal, Weibull y beta (continuas); Poisson y binomial negativa (conteo). Ceros técnicos → mitad del menor positivo.
- **Elección (D52, D56):** menor AIC entre las que KS (continuas) o chi² (conteos) no rechaza con α = 0,05; si todas se rechazan, bootstrap empírico. Si la elegida pone masa fuera de los límites de la variable, se reestima truncada y vuelve a pasar por la prueba.
- **Por jugador (D57):** la familia global con los parámetros del jugador si la global se aceptó y la del jugador no se rechaza; si no, bootstrap empírico del jugador; con < 10 observaciones, la global.
- **Resultado (3.6):** paramétricas `prof_sd_m` (gamma), cuadros de área (binomial negativa), déficit de su lado (exponencial truncada), ráfagas positivas (gamma) y las tres Bernoulli; bootstrap empírico `prof_media_m`, `vel_p90` y `pct_quieto`. Tablas: `distribuciones_global.csv` y `distribuciones_jugador.csv`.
- **Entrega para el sorteo (D55):** las distribuciones por variable y las filas completas de cada jugador × punto; cómo se sortea se decide en la fase 4.

## Registro de decisiones

Ordenado por sección del notebook. Toda decisión nueva se agrega aquí.

| # | Sección | Decisión | Fuente |
|---|---|---|---|
| D17 | General | Repositorio simplificado: un solo notebook (los dos de entendimiento integrados en la 1.2), un solo diccionario y `src/` en 4 módulos. Se eliminaron la guía PDF del 3-nov y el script del profesor para la traza. | Equipo (2026-11-07) |
| D18 | General | `CLAUDE.md` solo contiene reglas de comportamiento e hitos. Las reglas de datos, las decisiones y los pendientes viven en este README, y los anexos A y B del notebook enlazan aquí. | Equipo (2026-11-07) |
| D16 | 1.1 | Datos crudos en `data/raw/`; salidas en `data/interim/` y `data/processed/`. | Equipo (2026-11-06) |
| D19 | 1.1 | Datos versionados en el repositorio privado: los dos Excel y el video en Parquet por partido (27 archivos, 172 MB, el mayor de 13 MB; conversión exacta verificada). El CSV original (655 MB) supera el límite de GitHub y no se sube. Majo hace el push. | Equipo (2026-11-07) |
| D24 | 1.2.7 / 1.8 | Se descarta toda la pelota, incluida `player_hits_ball`: ~61 % de sus posiciones son relleno (50,5 %) o interpolación lineal (10,1 %), y `player_hits_ball` se deriva de esa posición (golpe = pelota a ~0,15 m) y coincide con picos de raqueta solo en el 41 % de los casos (22 % al azar). Los golpes son picos de velocidad de raqueta. | Equipo (2026-11-07) |
| D23 | 1.2.8 / 1.8 | Atípicos: velocidad > 8 m/s, aceleración > 10 m/s² (sobre la trayectoria suavizada) y angulares > 3.000 °/s; velocidades en píxeles por encima del percentil 99,5 de cada jugador. El valor pasa a NaN y se interpola en huecos de ≤5 frames; se guarda el % corregido. Diagnóstico en 1.2.8, corrección en 1.8. | Equipo (2026-11-07) |
| D4 | 1.3 / 1.5 | Jugadores sin formulario, con código entregado por el equipo: Alejandra Briget Deza Céspedes (407710), Juan Pablo Vanegas Herrera (356629) y Laura Sofía Rodríguez González (324764). Imputación con datos de su sexo: continuas con la media (más ruido N(0, sd) para las mujeres, con semilla fija y recorte al rango), ordinales con la mediana (la inferior si cae entre dos), categóricas con la moda. Llevan `imputado = True`. | Equipo (2026-11-06) |
| D8 | 1.3 | Códigos de 6 dígitos como texto. Jorge García: 99999 → `099999`. José Fabián García Camargo: 1075684546 → `107568` (primeros 6). | Equipo (2026-11-06) |
| D9 | 1.3 | Las variantes de nombre son el mismo jugador; nombre completo con dos apellidos. No se separan seguras y dudosas. | Equipo (2026-11-06) |
| D3 | 1.4 | Variable objetivo: ganador del punto (`PUNTO_EQUIPO_1`). | Equipo (2026-11-06) |
| D13 | 1.4 / 1.7 | `RIGTH` → `RIGHT`. Convención: BOTTOM en `y > 10` y RIGHT en `x > 5` para ambos equipos. | Verificado con los datos (99 %) |
| D1 | 1.5 | `TALLA` es el peso en kg; IMC = `TALLA` / (`ESTATURA` en m)². | Equipo (2026-11-06) |
| D5 | 1.5 | La fila sin nombre (152 cm, 48 kg) es Ana Catalina Suárez Caballero (337448); ya viene resuelta en el formulario. | Formulario actual (2026-11-06) |
| D10 | 1.5 | `FRECUENCIA_DEPORTE`: «0 veces» y «No practico deportes» se unen en el nivel 0. | Equipo (confirmada 2026-11-07) |
| D11 | 1.5 | Escalas ordinales (0 = menor) de nivel, tiempo jugando, experiencia, estado físico y frecuencia. `TIEMPO_JUGANDO_PADEL` se imputa como ordinal (mediana). | Análisis (ajuste a D4) |
| D2 | 1.6 / 1.8 | `racket_speed`, `wrist_speed` y `elbow_speed` en px/s; angulares en °/s. fps = 1/`duration` por punto; no se asume 60. | Código del pipeline del profesor |
| D6 | 1.6 | Los CSV de video se reciben ya generados; la base normalizada del profesor no se usa (es de otro torneo). | Revisión del repositorio del profesor (2026-11-06) |
| D7 | 1.6 | El partido 5, punto 10, es una copia del punto 9: el video se conserva como punto 9 y el punto 10 queda sin video. Regla general: conservar el de menor número y revisar con el equipo cada duplicado nuevo. | Equipo (2026-11-06) |
| D26 | 1.6 / 1.8 | Ningún partido recibe tratamiento especial: solo se aplican reglas generales que el notebook calcula (duplicados D7, identidad D12, `y` = `x` → `y` faltante). | Equipo (2026-11-07) |
| D30 | General | Solo se registran conclusiones demostrables con código u observación directa de los datos. Se retiraron la «homografía desplazada» (partidos 27 y 23-3) con su bandera `y_top_distorsionada`, el «remuestreo» del partido 32 con su bandera `fps_no_estandar` y la tabla de calidad partido por partido. | Equipo (2026-11-07) |
| D12 | 1.7 | Identidad de los tracks: por etiqueta si es coherente con la traza (≥50 % en la alineación y ≥80 % de concordancia); si no, por posición contra la traza (asignación codiciosa por filas × p(fondo) × p(lado)). Aplicado a los partidos 8, 17 y 34. | Equipo: «asignar según traza» (2026-11-06) |
| D14 | 1.7 | Los tracks `Player_N` y los que quedan sin casilla compatible se descartan de la identidad. | Análisis |
| D15 | 1.7 | Se excluyen los puntos de video sin traza (7-11, 11-12, 23-9 y 32-13): según la traza, esos partidos ya habían terminado. La numeración de puntos del video coincide con la de la traza. | Análisis; numeración confirmada por el equipo (2026-11-07) |
| D25 | 1.8 | Variables combinadas: `x`/`y` → velocidad, aceleración y desplazamiento; raqueta/muñeca/codo ÷ `body_scale` → `indice_vel_golpe`; raqueta sobre el hombro o muñeca sobre la cabeza → `juego_aereo`. | Equipo (2026-11-07) |
| D29 | 1.8 | Parámetros de la limpieza: margen de 0,5 m fuera de la cancha (recorte; más lejos, NaN); media móvil de 0,5 s; golpes = picos por encima del p90 del jugador en el punto, separados ≥ 0,4 s; rol = lado de la traza (drive/revés). | Equipo (2026-11-07) |
| D21 | 1.9 | Estilo atacante/defensivo: zona atacante = delante de la línea de saque (< 6,95 m de la red); atacante si pasa en ella al menos tanto tiempo como detrás. Se calcula por jugador × punto y por jugador (atacante si lo fue en ≥ 50 % de sus puntos). Se agrandó la zona (antes 0–3 m) porque son jugadores aficionados que casi no se acercan a la red: solo el 1,7 % del tiempo a menos de 3 m. Las tres secciones (red, transición, fondo) quedan como variables descriptivas. | Equipo (2026-11-07, revisada) |
| D22 | 1.9 | Ritmo de juego: duración del punto y golpes por segundo, por jugador y por pareja. | Equipo (2026-11-07) |
| D20 | 1.10 | Al modelo pareja contra pareja solo entran los 244 puntos con los 4 jugadores identificados. | Equipo (2026-11-07) |
| D27 | 1.10 / 2 | Selección de variables: en la fase 1, solo filtros que no usan el resultado (D33); Lasso (L1) y Ridge (L2) se aplican en la fase 2, dentro de la validación dejando un partido fuera, para no inflar las métricas. | Equipo (2026-11-07, revisada) |
| D31 | 1.9 | Definiciones jugador × punto: quieto < mediana global de velocidad; ráfaga = tramo ≥ 0,5 s sobre el p90 global; área = cuadros de 0,5 m pisados; variabilidad de ángulos = desviación estándar (circular para la raqueta); dirección de raqueta en 3 grupos (arriba, abajo, lateral); track < 2 s → todas NaN; pose solo si la cobertura es ≥ 50 %. | Equipo (2026-11-07) |
| D32 | 1.10 | Pareja × punto: suma y |diferencia| de cada variable del jugador (más razón mín/máx para velocidad p90, golpes por segundo y área); estilo = número de atacantes; sincronía = correlación de velocidad y de profundidad; `en_lado_mano` = diestro a la derecha o zurdo a la izquierda; una fila por punto con `e1_`, `e2_` y `dif_` = equipo 1 − equipo 2. | Equipo (2026-11-07) |
| D33 | 1.10 | Faltantes: quedan NaN y se imputan en la fase 2 con la mediana de entrenamiento. Filtros: faltantes > 20 %, valor dominante ≥ 95 %, Pearson |r| ≥ 0,85 (se conserva la de menos faltantes; si empatan, la `dif_`). Se entregan todas las que pasan. | Equipo (2026-11-07) |
| D34 | 1.11 | Variables para las fases 2 a 4: 19 `dif_` que se reconstruyen desde cada jugador (estilo y posición 5, movimiento 6, nivel y atributos fijos 8). Se quita `dif_suma_mujer`: r = −0,85 con `dif_suma_estatura_cm` y VIF 10,3 (estatura 12,8); sin ella el VIF máximo es 4,1. Fuera: pose y golpes, cohesión y sincronía, duración del punto, `pose_coverage`, las de poca variación y las repetidas. | Equipo (2026-11-16) |
| D35 | 2.1 | Partición como en los notebooks del profesor (train/test 80/20 con `random_state=0` y validación cruzada en train), pero por partido: test = partidos 5, 10, 18, 26 y 29 (52 puntos); train = 20 partidos (192 puntos) con un pliegue por partido. | Equipo (2026-11-16) |
| D36 | 2.2 | Evaluación común (`src/modelo.py`): imputación por mediana dentro de cada pliegue y AUC calculado dentro de cada partido validado y promediado (como `cross_val_score(scoring='roc_auc')`). Con todas las predicciones juntas el naive da AUC 0,271 en vez de 0,5 (correlación −0,98 entre lo real de cada partido y su predicción). Log-loss, Brier, accuracy y F1 con todas las predicciones juntas. Los partidos 28 y 35 no tienen AUC (una sola clase). | Equipo (2026-11-16) |
| D37 | 2.3 | Interpretación de la logística con `statsmodels`: variables imputadas con la mediana del train y estandarizadas con el train (odds ratio por una desviación estándar), y errores estándar agrupados por partido (`cov_type='cluster'`). | Análisis (2026-11-16) |
| D38 | 2.4 | La penalización de LASSO/Ridge (`C`) y la profundidad del árbol se eligen por el log-loss de la validación cruzada por partido; el AUC por pliegue se muestra al lado. `min_samples_leaf=10` fijo en el árbol. Desde `scikit-learn` 1.8 el castigo se elige con `l1_ratio` (1 = LASSO, 0 = Ridge); por eso `requirements.txt` pide `scikit-learn>=1.8`. | Análisis (2026-11-16) |
| D39 | 2.5 | Todas las combinaciones se comparan con los 20 pliegues por partido de la 2.1 y se eligen por log-loss de las predicciones reunidas (D38). Se usa una función propia (`buscar`) y no `GridSearchCV`: este promedia el log-loss de cada pliegue, mientras que aquí se calcula sobre todas las predicciones juntas, igual que en la 2.2 (D36). La grilla SVM se evalúa completa; la búsqueda aleatoria se reserva para el espacio más amplio del Random Forest. No se usa optimización bayesiana: en el Random Forest la superficie observada es plana (las ocho mejores combinaciones de la grilla y de la búsqueda aleatoria quedan entre log-loss 0,651 y 0,655), así que una búsqueda más inteligente tiene poco dónde mejorar; la grilla SVM, con solo 20 combinaciones, se evalúa completa (sus ocho mejores van de 0,652 a 0,674). **Limitación:** los hiperparámetros se eligen con los mismos pliegues con los que se reporta (sin validación anidada), así que la métrica del mejor modelo es algo optimista. El bootstrap por partido del final resume la incertidumbre entre las predicciones observadas, pero no elimina ese sesgo de selección. | Equipo (2026-11-20) |
| D40 | 2.5 | Se comparan 100, 300 y 500 árboles, junto con profundidad, tamaño mínimo de hoja, variables candidatas y fracción de observaciones por árbol. El tamaño mínimo de hoja nunca baja de 3. | Equipo (2026-11-20) |
| D41 | 2.5 | El escalado se ajusta dentro de cada pliegue. Se comparan kernels `rbf` y `linear`, `C` y `gamma`. Las probabilidades se calibran con una regresión logística sobre puntajes *out-of-fold*; también esa calibración separa partidos, nunca puntos al azar. Para `svm_lasso`, los hiperparámetros se eligieron con las 19 variables y se reutilizaron tal cual con la selección de LASSO dentro de cada pliegue; por eso la comparación con `svm` es aproximada y, si esta variante pasa a la 2.7, convendría reajustarlos. | Equipo (2026-11-20) |
| D42 | 2.6 | Bagging de árboles con `n_estimators=200` fijo; se ajustan profundidad, hoja mínima y `max_samples` (grilla de 27). El OOB se reporta pero no se usa para elegir (los árboles que no vieron un punto sí vieron otros de su partido). Importancia = promedio de la importancia de cada árbol. | Equipo (2026-11-20) |
| D43 | 2.6 | Boosting con árboles poco profundos y *learning rate* bajo. AdaBoost: grilla completa (profundidad 1–2, 25 a 200 árboles, *learning rate* 0,01 a 1). Gradient Boosting: búsqueda aleatoria de 40 combinaciones (profundidad 1–3, *learning rate* 0,01 a 0,1, `subsample`, hoja mínima 5–20). Elección por log-loss, con la limitación de D39. | Equipo (2026-11-20) |
| D44 | 2.6 | XGBoost es opcional: entra solo si `xgboost` está instalado (no está en `requirements.txt`). En la ejecución del 2026-11-21 no estaba instalado y se omitió. | Equipo (2026-11-20) |
| D45 | 2.6 | Red neuronal `MLPClassifier` pequeña (1 o 2 capas de 4 a 16 neuronas), escalado dentro del pliegue y penalización `alpha` de 0,01 a 10; sin `early_stopping`, porque separaría una validación al azar que mezcla puntos del mismo partido. | Equipo (2026-11-20) |
| D46 | 2.7 | Ensamblaje de Gradient Boosting, LASSO y RF (los candidatos de la 2.6). Promedio simple de probabilidades (*soft voting*) y voto ponderado con pesos ∝ 1 − log-loss (como el 1 − error del notebook 06), calculados en cada pliegue solo con los otros 19 partidos. El voto mayoritario 0/1 se reporta solo en accuracy y F1 (no da probabilidad para la fase 4). No se reentrena: la probabilidad de validación de cada modelo ya viene de su modelo entrenado sin ese partido. | Equipo (2026-11-21) |
| D47 | 2.7 | Como el promedio no superó al Gradient Boosting, se probaron otras combinaciones aprendidas solo con los otros 19 partidos de cada pliegue (pesos óptimos con 3 y con 9 modelos, selección codiciosa de Caruana, stacking con logística, promedio de logits, promedio de 9 y recalibración). Se queda la de menor log-loss en validación: pesos óptimos con 3 modelos (`ens_optimo`, `mod.pesos_optimos`): 0,745 Gradient Boosting + 0,255 LASSO + 0 RF; log-loss 0,631, AUC 0,616. Limitación: las predicciones de segundo nivel vienen de modelos que vieron el partido validado. | Equipo (2026-11-21) |
| D48 | 2.8 | Regla fija antes del test: el modelo final es `ens_optimo`, elegido con la validación por partido. El test (5 partidos, 52 puntos) se usa una sola vez, se reporta para todos los candidatos y no se usa para elegir. Bootstrap del test por partido (2.000 remuestreos, semilla 2026). | Equipo (2026-11-21) |
| D49 | 2.8 | El modelo de la fase 4 (`modelo_fase4`) es `ens_optimo` reentrenado con los 244 puntos de los 25 partidos, con los mismos hiperparámetros y pesos. La métrica reportada sigue siendo la del test. | Equipo (2026-11-21) |
| D50 | 3 | Datos y nivel: todos los jugador × punto con video (1.015; 979 con profundidad y movimiento), no solo los 244 puntos del modelo. Ajuste global y por jugador; por jugador solo los componentes con ≥ 10 observaciones, si no la global. | Equipo (2026-11-22) |
| D51 | 3.1 | Orden temporal `(partido, punto, equipo, lado)`, suponiendo que el número de partido sigue el orden en que se jugaron; media móvil de 30 observaciones; estacionaria si ADF rechaza (p < 0,05) y KPSS no (p ≥ 0,05); ACF de rezago 1 dentro de cada jugador. | Equipo (2026-11-22) |
| D52 | 3.4 | Elección: menor AIC entre las candidatas que la prueba de bondad no rechaza (KS para continuas, chi² para conteos, α = 0,05). Si todas se rechazan, se reporta la de menor AIC y la recomendación es el bootstrap empírico. | Equipo (2026-11-22) |
| D53 | 3.4 | Componentes: `pct_en_su_lado` = Bernoulli P(= 100) + déficit 100 − x; `rafagas_por_min` = Bernoulli P(= 0) + positivas (valla); `area_m2` = conteo de cuadros extra (área / 0,25 − 1), que es lo que usa la fase 4 (como continua solo se compara); `atacante` = Bernoulli. | Equipo (2026-11-22) |
| D54 | 3.4 | Candidatas: normal, uniforme, triangular, exponencial, gamma, Erlang, chi², lognormal, Weibull y beta (continuas); Poisson y binomial negativa (conteos). MLE con origen en 0 (gamma, exponencial, lognormal, Weibull), Erlang con forma entera, chi² con escala 1, uniforme y triangular entre mínimo y máximo, beta solo con límites conocidos (Smithson y Verkuilen). Ceros técnicos → mitad del menor positivo. | Equipo (2026-11-22) |
| D55 | 3 / 4 | La fase 3 entrega las distribuciones por variable y las filas completas de cada jugador × punto (para bootstrap por fila). Si se sortea por variable, por fila o con cópula se decide en la fase 4. | Equipo (2026-11-22) |
| D56 | 3.4 | Si la distribución elegida pone masa fuera de los límites de la variable, se reestima por MLE truncada a esos límites y la truncada vuelve a pasar por D52. Aplicado: `pct_quieto` (Weibull con 4,2 % > 100; truncada KS p = 0,031 → bootstrap empírico) y déficit de su lado (exponencial con 0,43 % > 100; truncada KS p = 0,22 → paramétrica). | Equipo (2026-11-22) |
| D57 | 3.5 | Por jugador: la familia elegida en la global con los parámetros del jugador donde la global se acepta y la prueba del jugador no la rechaza; bootstrap empírico del jugador donde se rechaza; la global con < 10 observaciones (D50). | Equipo (2026-11-22) |
| D28 | 2 | Comparar un modelo base con modelos avanzados, hacer un ensamblaje de modelos y usar bootstrap para medir la fiabilidad. En 2.5, RF (AUC 0,601; log-loss 0,651; Brier 0,229), LASSO (0,579; 0,651; 0,229) y SVM-LASSO (0,586; 0,652; 0,229) quedan como candidatos para el ensamblaje. En 2.6, el Gradient Boosting (`max_depth=1`, 200 árboles, *learning rate* 0,05) obtiene el menor log-loss (0,630; Brier 0,219; AUC 0,602), pero su IC95 % del bootstrap contra LASSO incluye cero ([−0,056; 0,015]); Bagging (0,652), red neuronal (0,656) y AdaBoost (0,662) no mejoran a LASSO. Candidatos para la 2.7: Gradient Boosting, LASSO y RF. En 2.7 el ensamblaje de los tres (promedio y ponderado dan lo mismo) obtiene log-loss 0,634 y Brier 0,222: mejora a LASSO y RF (IC95 % excluyen cero) pero no al Gradient Boosting solo (+0,004 [−0,019; 0,027]); los tres modelos tienen errores con correlación 0,96–0,98. Con pesos óptimos (D47) el ensamblaje llega a 0,631 y es el modelo final (D48). En el test de la 2.8 todos los modelos quedan peor que el naive (`ens_optimo` 0,766 contra 0,674; IC95 % de la diferencia [0,039; 0,152]) y con AUC < 0,5. | Equipo / análisis (2026-11-16) |

## Supuestos y pendientes

| Tema | Detalle | Acción |
|---|---|---|
| Variables finales (D27) | LASSO conserva 7 de las 19 (nivel sumado, % quieto, IMC, desbalance de nivel, velocidad p90, diferencia de profundidad entre compañeros y `prof_sd`), por debajo de la meta de 10–15, y no deja ninguna de estilo atacante/defensivo. Ridge conserva las 19 con un log-loss casi igual (0,655 vs 0,651). | Decidir con el equipo: las 7 de LASSO, forzar 10–15 o usar Ridge con las 19 |
| Resultado en test (D48) | En los 5 partidos del test ningún modelo supera al naive: `ens_optimo` 0,766 contra 0,674 en log-loss (IC95 % de la diferencia [0,039; 0,152]) y AUC < 0,5 en todos. La mejora de validación (−0,066) no se repitió. El test ya se usó: cualquier cambio de modelo no puede volver a medirse con él. | Decidir con el equipo cómo sigue la fase 4: (a) usar `modelo_fase4` con esta advertencia; (b) usar un modelo más simple, sabiendo que la elección ya no tiene test independiente; (c) estimar el desempeño con validación por partido sobre los 25 partidos y reportarlo junto al test |
| Simetría del modelo (2.8) | Al intercambiar las parejas, p + p' da en promedio 0,93 (entre 0,62 y 1,31), y con dos parejas iguales el modelo final le da 0,602 al equipo 1. | Decidir antes de la fase 4 cómo comparar parejas, por ejemplo promediando p(A contra B) y 1 − p(B contra A) |
| Cómo sortear en la fase 4 (D55) | Las variables de movimiento del mismo jugador y punto tienen correlación de hasta 0,83 (3.3) y los compañeros se parecen (hasta 0,66; 3.2). Sortear cada variable por separado rompe esas relaciones. | Decidir en la fase 4: variables independientes, fila completa del jugador o cópula |
| Dependencia entre puntos (3.1) | La profundidad y el estilo de un jugador se repiten de un punto al siguiente (ACF de rezago 1 ≈ 0,5 dentro de cada jugador). | Decidir en la fase 4 si se sortea cada punto de forma independiente |
| Cambio de nivel en el torneo (3.1) | `area_m2`, `pct_quieto` y `rafagas_por_min` no son estacionarias (KPSS p ≤ 0,014): el área baja de 3,78 a 2,42 m² y el % quieto sube de 45,6 a 55,7 entre el primer y el último tercio de partidos. | Decidir en la fase 4 si se sortea de todo el torneo |
| Búsquedas en el borde (2.6) | En el Gradient Boosting (`min_samples_leaf=20`) y en la red (`alpha=10`), el mejor valor quedó en el borde de la grilla. XGBoost no se evaluó (D44). | Opcional; ya no puede medirse con el test |
| Datos faltantes | 15 puntos de la traza sin video. Los 12 partidos programados sin traza ni video aparecen en `Partidos_Jugados` con 0 puntos. | Confirmar si se jugaron; si llegan datos, se vuelve a correr la fase 1 |
| Apellidos | 5 jugadores con un solo apellido en todas las fuentes (Jeronimo Coy, Jorge García, Juliana Gómez, Marcela Velásquez, Miguel Delgado). | Opcional: buscar en Teams |

## Origen

El proyecto parte del material del profesor (`padel-vision-analytics-light`). De ese material solo se conservó lo necesario para la analítica: la documentación de las columnas del CSV de video (reescrita en `docs/diccionario.md`), el mapa de keypoints y las dimensiones de la cancha. No se usa el pipeline de visión (YOLO, pose, tracking), porque los CSV de video se reciben ya generados.
