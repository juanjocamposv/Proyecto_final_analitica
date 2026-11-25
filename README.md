# Proyecto final de analítica: parejas ideales de pádel

Encontrar las parejas que maximizan la probabilidad de ganar el torneo según su estilo de juego. Se parte de tres fuentes: datos de video (pose y posición por frame), la traza de partidos y el formulario de jugadores.

**¿Por dónde empezar?** El trabajo (código, resultados y reporte, por fases) está en [notebooks/Proyecto_Padel.ipynb](notebooks/Proyecto_Padel.ipynb). El notebook anterior, `Proyecto_Parejas_Padel.ipynb`, queda solo como consulta: ya se migró de él lo que servía (D1). Este README explica los archivos del repositorio y concentra las reglas de datos, el [registro de decisiones](#registro-de-decisiones) y los [pendientes](#supuestos-y-pendientes).

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
| 1. Manejo de datos | Entendimiento de las tres fuentes; faltantes, atípicos y tratamiento; 14 variables preliminares jugador × punto en 3 grupos; correlación (quedan 12); tabla equipo × punto (1.1–1.5) | Hecha en el notebook nuevo (D1–D27) |
| 2. Predictiva | Partición y evaluación común; naive y logística por pasos; otros modelos del curso; ensamblaje y test (2.1–2.4) | Hecha (D28–D39); rehecha con AUC y búsqueda bayesiana el 2026-11-24 |
| 3. Distribuciones | Exploración; ajuste con todos los jugadores (10 candidatas, KS y AIC) y por jugador de `atacante` y `vel_media_mps` (3.1–3.3) | Hecha (D40–D44) |
| 4. Simulación | Del punto al partido; revisión con los partidos reales; todos contra todos de las 435 parejas con el ensamblaje (D38) y las distribuciones de la fase 3; incertidumbre por remuestreo de partidos; parejas ideales según su estilo (4.1–4.5) | Hecha (D45–D50) |

## Archivos del repositorio

```
README.md                         Este archivo: archivos, reglas de datos, decisiones y pendientes
CLAUDE.md                         Reglas de trabajo para el asistente de IA (Claude Code) e hitos
requirements.txt                  Librerías de Python necesarias
.gitignore                        Lo que no se sube a git
notebooks/
  Proyecto_Padel.ipynb            Notebook de trabajo: todas las fases y el reporte (D1)
  Proyecto_Parejas_Padel.ipynb    Notebook anterior: solo consulta (ya no se migra nada)
  referencias/                    Notebooks del profesor que se replican en la fase 2 (solo consulta)
src/                              Funciones del pipeline que usa el notebook
  __init__.py                     Marca src/ como paquete y lista sus módulos
  datos.py                        Rutas, constantes de cancha, lectura de fuentes, utilidades
  identidades.py                  Nombres canónicos, alias y códigos de estudiante
  fuentes.py                      Limpieza de la traza y del formulario
  video.py                        Identidad, faltantes, pelota y limpieza de posición y velocidad
  variables.py                    Variables jugador × punto, correlación y tabla equipo × punto
  modelo.py                       Fase 2: partición, evaluación, pasos, búsqueda bayesiana, ensamblaje y test
  distribuciones.py               Fase 3: exploración, ajuste de distribuciones y tabla por jugador
  simulacion.py                   Fase 4: del punto al partido, todos contra todos y remuestreo
docs/
  diccionario.md                  Diccionario de todas las columnas (fuentes y tablas derivadas)
  pose_keypoints_map.jpg          Imagen de los puntos de pose que usa el diccionario
data/
  raw/                            Datos de entrada (versionados, D5)
    Traza_Partidos_202602.xlsx
    Datos_Jugadores_Padel_202602.xlsx
    video/partido_XX.parquet      Video en Parquet, un archivo por partido (27)
  interim/                        Tablas que genera el notebook (no se suben)
  processed/                      Tabla del modelo, configuración del ensamblaje y ranking (no se suben)
```

### Qué es y qué hace cada archivo

| Archivo | Qué es | Qué hace / cuándo usarlo |
|---|---|---|
| `notebooks/Proyecto_Padel.ipynb` | El documento de trabajo del proyecto (D1). | Se ejecuta de principio a fin (unos 37 min: casi todo es la búsqueda bayesiana de la 2.3 y unos 7 min la fase 4). Cada sección tiene Objetivo → Decisiones → Código → Resultados, con texto corto como en los notebooks del profesor. Fase 1: 1.1 entendimiento, 1.2 faltantes, atípicos y tratamiento, 1.3 variables preliminares, 1.4 correlación y 1.5 base unificada. Fase 2: 2.1 partición y evaluación, 2.2 naive y logística por pasos, 2.3 otros modelos del curso y 2.4 ensamblaje y test. Fase 3: 3.1 exploración, 3.2 ajuste con todos los jugadores y 3.3 ajuste por jugador y tabla para la fase 4. Fase 4: 4.1 del punto al partido, 4.2 revisión con los partidos reales, 4.3 ranking, 4.4 incertidumbre y 4.5 parejas ideales según su estilo (unos 7 min; para correrla sola se necesitan las tablas de `data/` y `data/processed/ensamblaje.joblib`). |
| `notebooks/Proyecto_Parejas_Padel.ipynb` | El notebook anterior. | Solo consulta: ya se migró lo que servía. Sus fases y sus tablas ya no corresponden a `src/` y no se ejecuta. |
| `notebooks/referencias/` | Notebooks de clase del profesor. | Solo consulta: la fase 2 replica su estructura y su código. No se ejecutan ni forman parte del pipeline. |
| `CLAUDE.md` | Instrucciones para el asistente de IA. | Cómo debe trabajar (preguntar antes de decidir, verificar, mantener el repositorio simple) y los hitos del proyecto. No tiene decisiones concretas: remite a este README. |
| `src/datos.py` | La base del pipeline. | Rutas de `data/`, semilla (2026), medidas de la cancha, lectura de las fuentes (`leer_video`, `leer_traza`, `leer_formulario`, `iterar_video`, `tipos_video`), conversión del CSV de video a Parquet (`convertir_video_a_parquet`) y utilidades de reporte y escritura (`resumen_tabla`, `comparar`, `perfil_columnas`, `guardar_tabla`). |
| `src/identidades.py` | La tabla de equivalencias de jugadores. | Lleva cualquier nombre de cualquier fuente a su nombre canónico y a su código de 6 dígitos. Aquí se agregan alias nuevos, jugadores sin formulario y correcciones de código. |
| `src/fuentes.py` | La limpieza de las dos fuentes pequeñas. | Traza: controles de consistencia, erratas, códigos y formato largo jugador × punto. Formulario: variables usables, escalas ordinales, imputación de los jugadores sin formulario e IMC. |
| `src/video.py` | El tratamiento del video. | Calidad por punto e identidad de cada track según la traza; % de vacíos por familia de columnas (`faltantes_por_familia`); evidencia de la pelota; limpieza frame × jugador de la posición y la velocidad (recorte a la cancha, saltos del tracking, suavizado y límites físicos). |
| `src/variables.py` | Las variables del modelo. | Los 3 grupos (`GRUPOS`), las 4 del primer modelo (`MODELO_1`) y las de one-hot (`ONE_HOT`); `variables_video` (respecto de su fondo, D22) y `variables_formulario` (con el one-hot, 1.3, D21); `efecto_fondo` (mismo jugador en TOP y BOTTOM); `pares_correlacionados` y `quitar_repetidas` (1.4, D24); `tabla_equipos` (1.5, D26, D27); `columnas_jugador`, `variable_de` y `columnas` (pasan de variables a columnas y al revés). |
| `src/modelo.py` | La fase 2. | `opciones_test` y `elegir_test` (test 80/20 por partido, D29); `Simetrico` (cualquier clasificador con p(A) + p(B) = 1) y `predecir` (validación dejando un partido fuera con `cross_val_predict`), `metricas` (AUC, accuracy y F1), `remuestreos`, `resumen` y `diferencia_auc` (intervalos por bootstrap de partidos, D30); `seleccion_por_pasos` (D31) y `coeficientes`; `buscar` (`BayesSearchCV` por partido con AUC) y `ajustar` (D32–D33); `pesos_accuracy`, `ensamblar` y `predecir_test` (D37, D39). |
| `src/distribuciones.py` | La fase 3. | `observaciones` (D41), `por_jugador`, `diferencias_entre_jugadores`, `dependencias` y `por_tercio` (3.1); `ajustar`, `comparar_candidatas` y `elegir` (10 candidatas, KS y AIC, D43); `ajustar_por_jugador` (D42, D44) y `distribucion_vel` (reconstruye la distribución de un jugador desde la tabla guardada, para la fase 4). |
| `src/simulacion.py` | La fase 4. | `guardar_ensamblaje`, `leer_ensamblaje` y `Ensamblaje` (la configuración de la 2.4 y el voto ponderado, D45); `prob_partido` y `simular_partidos` (4.1, D47); `tabla_jugadores`, `sortear`, `prob_punto` y `partidos_reales` (D46, 4.2); `parejas`, `enfrentamientos`, `todos_contra_todos`, `intervalo_t` y `caracteristicas_parejas` (4.3, D48); `replica_remuestreo` y `resumen_remuestreo` (4.4, D49). |
| `src/__init__.py` | Archivo técnico de Python. | Permite hacer `from src import ...` desde el notebook. |
| `docs/diccionario.md` | El diccionario de datos. | Parte A: cada columna de las fuentes crudas (unidad, cómo se genera y uso). Parte B: cada columna de las tablas que genera el notebook. |
| `docs/pose_keypoints_map.jpg` | Imagen de referencia. | Puntos del cuerpo que detecta el modelo de pose (hombro, codo, muñeca, raqueta y cabeza). |
| `requirements.txt` | Lista de librerías. | `pip install -r requirements.txt`. |
| `.gitignore` | Configuración de git. | Sube los Excel y el Parquet del video; deja fuera el CSV original, las tablas generadas, los cachés y los entornos. |

### Tablas que genera el notebook (`data/interim/`)

| Tabla | Una fila por | Contenido |
|---|---|---|
| `equivalencias_nombres.csv` | nombre de cada fuente | Nombre original → nombre canónico → código |
| `jugadores.csv` | jugador (32) | Variables fijas del formulario, codificadas e imputadas, más el IMC |
| `participaciones.csv` | jugador × punto (1.132) | Equipo, fondo, lado, compañero, si ganó el punto, su track de video y las 14 variables preliminares (1.3) |
| `identidades_video.csv` | track de video (1.201) | A qué jugador corresponde cada track, cómo se asignó y si se conserva |
| `puntos.csv` | punto (287) | Ganador, jugadores, estado del video y banderas de calidad |
| `video_limpio.parquet` | frame × jugador (705.888) | Video limpio: posición suavizada, distancia a la red, velocidad y aceleración (1.2) |
| `distribuciones_jugador.csv` | jugador (30) | Distribución de `atacante` y `vel_media_mps` de cada jugador para la fase 4 (3.3) |
| `processed/ensamblaje.joblib` | modelo (4) | Configuración del ensamblaje que guarda la 2.4: cada modelo sin entrenar con sus hiperparámetros, sus variables y su peso (D45) |
| `processed/ranking_parejas.csv` | pareja (435) | Puntaje de cada pareja en el todos contra todos, su error de simulación, sus intervalos del remuestreo y su estilo (4.3–4.5) |
| `processed/tabla_modelo.csv` | equipo × punto (448) | Tabla para el modelo (1.5): 224 puntos, `gano` y el promedio de las 12 variables (16 columnas con el one-hot) del equipo (`propio_`) y del rival (`rival_`) |

El detalle de cada columna está en la parte B de [docs/diccionario.md](docs/diccionario.md).

## Datos e instalación

Los datos están versionados en este repositorio, que es privado (D5). Basta con clonarlo o hacer `git pull`.

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
  - 244 tienen a los 4 jugadores identificados (D25); de ellos, 224 tienen las variables de video de los 4 y entran al modelo (D26).

### Casos especiales del video

No hay tratamientos por partido (D3, D2). Los casos especiales se detectan con reglas generales que el notebook calcula y muestra:
- **Duplicados exactos** de un punto dentro del partido: se conserva el de menor número (D16). Hoy aplica al partido 5, punto 10.
- **`y` igual a `x`** en más del 95 % de las filas de un punto: la `y` se marca faltante. Hoy aplica al partido 17.
- **Etiquetas de nombre que no concuerdan con la traza:** la identidad se asigna por posición (D17). Hoy aplica a los partidos 8 (etiquetas de jugadores que no están en su alineación), 17 (`Jugador 1–4`) y 34 (concordancia del 46 %; el track etiquetado «David Neisa» queda en la casilla de Alejandra Deza).
- **Puntos de video sin traza:** se excluyen (D19).

### Identidad de los jugadores

- **Nombres (D11):** las variantes de un nombre son el mismo jugador. El nombre canónico es el completo, con dos apellidos cuando alguna fuente lo tiene (`NOMBRES_CANONICOS`).
- **Códigos (D9, D10):** todos tienen 6 dígitos y se guardan como texto (`CORRECCIONES_CODIGO`, `JUGADORES_SIN_FORMULARIO`).
- **Tracks del video (D17, D18):**
  - Se usa la etiqueta del video si el partido tiene ≥50 % de filas con nombres de la alineación y ≥80 % de concordancia con la casilla (fondo, lado) de la traza.
  - Si no, la identidad se asigna por posición contra la traza.
  - Los `Player_N` y los tracks sin casilla compatible se descartan.
- **Búsqueda en Teams (proceso recurrente):** todo jugador que no cruce (con videos, partidos o formularios nuevos) se lista en la sección 1.2 y se busca en Teams. Con el nombre y el código confirmados, se agrega a `src/identidades.py` y se vuelve a correr la fase 1. Hoy no hay jugadores sin cruzar.

### Limpieza del video (sección 1.2)

- **Solo se limpian los tracks conservados** en `identidades_video.csv`, y solo la posición y la velocidad: no se usa ninguna columna de pose ni de raqueta (D15).
- **Pelota (D14):** se descartan todas sus columnas, incluida `player_hits_ball`. El 50,5 % de sus posiciones es relleno y el 10,1 % interpolación lineal.
- **Fuera de la cancha (D20):** las posiciones hasta 0,5 m afuera se recortan al borde; las más lejanas pasan a NaN.
- **Suavizado (D20):** media móvil centrada de 0,5 s de `x` e `y`. La velocidad y la aceleración se recalculan sobre esa trayectoria, con el fps de cada punto. Las columnas originales no sirven: la posición cambia a saltos de 4,9 cm, así que la velocidad cruda toma casi solo valores como 0 y 2,93 m/s.
- **Atípicos (D20):** velocidad > 8 m/s o aceleración > 10 m/s² pasan a NaN y se interpolan si el hueco es de ≤5 frames. Primero se quitan las posiciones con saltos del tracking y luego se recalcula todo.
- **Descartadas:** pose y raqueta (D15), pelota (D14), `duration` (solo para el fps), `total_frames`, `video_duration`, `prev_*`, `player_displacement`, `distance_player_to_teammate_m`, `zone` (no es un rol), `schema_version` y `*_implausible`.

### Formulario y traza

- **Fuga de datos:** `Puntos_Total`, `Partidos_Jugados` y `Puntos_Por_Partido` nunca son predictoras; solo validan la traza (`Puntos_Total` coincide en 27 de 27 jugadores).
- **Usables:** `ESTATURA`, `TALLA` (peso, D8), `MANO_DOMINANTE` (solo 2 zurdos), `NIVEL_ACTUAL_PADEL`, `TIEMPO_JUGANDO_PADEL`, `EXPERIENCIA_PADEL`, `PRACTICA_OTRO_DEPORTE_RAQUETA`, `ESTADO_FISICO`, `FRECUENCIA_DEPORTE`, `EDAD`, `GENERO` y `LESIONES_LY`. Las escalas ordinales están en `ORDINALES` (D12).
- **Descartadas:** ciudad, programa, semestre, razón y objetivo de participación, interés y clases de pádel, desayuno, género musical y comentarios. También `RITMO_CARDIACO`: tiene pocos datos (12 de 29) y no interesa (confirmado por el equipo el 2026-11-07).
- **Fondo:** el equipo 1 siempre está en BOTTOM. Por eso las variables de video se miden respecto de su fondo (D22).
- **Traza:** pasa todos los controles (un ganador por punto, marcador igual al acumulado, puntos consecutivos, lados fijos por partido). El equipo 1 gana el 43,1 % de los puntos.

### Variables (secciones 1.3 a 1.5)

Granularidad: **frame × jugador → jugador × punto → equipo × punto**.

- **14 variables preliminares por jugador × punto en 3 grupos (D21):**
  - **Nivel de juego** (formulario): `nivel_padel` (primera vez, principiante o intermedio/avanzado), `tiempo_jugando_padel` (0 = menos de 6 meses, 1 = 6 meses o más), `experiencia_padel` y `practica_otra_raqueta` (0/1).
  - **Tipo de juego** (video, respecto de su fondo, D22): `atacante`, `dist_red_m`, `vel_media_mps` y `pct_quieto`.
  - **Atributos físicos** (formulario): `estatura_cm`, `imc`, `edad`, `estado_fisico` (malo/regular, bueno o excelente), `frecuencia_deporte` (0, 1–2, 3–4 o más de 4 veces por semana) y `mujer`.
  - One-hot (por recomendación del profesor): `nivel_padel`, `estado_fisico` y `frecuencia_deporte` pasan a columnas 0/1 con `pd.get_dummies(drop_first=True)` (`nivel_padel_1`, `nivel_padel_2`, `estado_fisico_1`, `estado_fisico_2`, `frecuencia_deporte_1` a `_3`); la primera categoría es la base. Las categorías de una sola persona se unen a la vecina. Track < 2 s → sin variables de video (D21).
- **Respecto de su fondo (D22):** atacante = pasa al menos la mitad del punto más cerca de la red que la mediana de su fondo; quieto = por debajo de la mediana de velocidad de su fondo; a `dist_red_m` y `vel_media_mps` se les resta la mediana de su fondo.
- **No se usan (D23, D15):** pose y raqueta, velocidades en píxeles, aceleración, columnas `_x`/`_y` y `body_scale` como variables, pelota, puntos ganados, y sumas, diferencias o razones entre jugadores.
- **Correlación (D24):** si |r de Spearman| ≥ 0,8, se deja la del modelo 1. Salen `pct_quieto` y `dist_red_m`: quedan 12 variables.
- **Equipo × punto (D27, D26):** promedio de los dos jugadores; dos filas por punto con `propio_` y `rival_`; objetivo `gano`. Se eliminan los puntos con algún jugador sin variables de video: quedan 224 puntos de 24 partidos, sin imputar nada.
- **Orden de entrada al modelo (D31):** modelo 1 = `nivel_padel`, `atacante`, `vel_media_mps`, `estatura_cm`; luego el resto de nivel de juego y de atributos físicos (tras D24, tipo de juego no agrega variables). Un grupo entra solo si el intervalo del 95 % de la mejora de AUC queda por encima de 0; si no, sus variables se prueban una por una.

### Modelado (fase 2)

- **Objetivo:** `gano` del equipo de cada fila (D7, D27). La probabilidad de ganar el partido a 7 puntos se deriva de p(ganar el punto) en la fase 4.
- **Estructura (D28):** 2.1 partición y validación; 2.2 naive y logística por pasos; 2.3 LASSO/Ridge, árbol, Random Forest, SVM, bagging, boosting y red neuronal; 2.4 ensamblaje y comparación final en test.
- **Partición (D29):** test = partidos 7, 14, 24, 29 y 39 (53 puntos), usado una sola vez al final; train = los otros 19 (171 puntos). Nunca se mezclan puntos de un mismo partido.
- **Evaluación (D30):** validación dejando un partido fuera en el train; predicciones simetrizadas; AUC (principal), accuracy y F1, con intervalos del 95 % por bootstrap de partidos.
- **Referencia mínima:** el naive. Con las filas espejo predice 0,5, así que su AUC es 0,5 y su accuracy 50 %.
- **Selección (D31, D32):** pasos desde las 4 variables del modelo 1; un grupo o variable entra solo si el intervalo del 95 % de la mejora de AUC queda por encima de 0. La logística y cada modelo de la 2.3 lo hacen por su cuenta.
- **Hiperparámetros (D33):** búsqueda bayesiana (`BayesSearchCV`) como el profesor, 40 evaluaciones por modelo, con un partido fuera por pliegue y AUC; se buscan con las 4 variables del modelo 1.
- **Ensamblaje (D36):** entran los 4 mejores modelos de la 2.3 por AUC de validación.
- **Voto ponderado (D37):** α = accuracy / suma de las accuracy (como el notebook 06, con la validación por partido en vez del out-of-bag), sobre las probabilidades.
- **Modelo final (D38) y test (D39):** el de mayor AUC de validación pasa a la fase 4; todos los modelos se miden una sola vez en el test, con intervalo bootstrap por partido.
- Las parejas casi no se repiten (30 jugadores): el modelo aprende de características de la pareja, no de identidades.

### Distribuciones (fase 3)

- **Qué se sortea (D40):** solo `atacante` y `vel_media_mps`, las variables del ensamblaje que cambian de punto a punto. `nivel_padel` y `estatura_cm` son fijas de cada jugador.
- **Datos (D41):** las 979 filas jugador × punto con variables de video, de 30 jugadores (de 7 a 57 puntos por jugador).
- **Por jugador (D42):** cada jugador con su distribución; con menos de 10 puntos (Isabella Arenas y Jerónimo Coy, 7 cada uno), la global.
- **`atacante` (D43, D44):** Bernoulli con p = proporción de puntos como atacante (0,473 con todos; de 0,03 a 1,0 por jugador).
- **`vel_media_mps` (D43, D44):** 10 candidatas por máxima verosimilitud, con desplazamiento y escala; gana la de menor AIC entre las que KS no rechaza (α = 0,05). Gana la gamma (forma 2,92; desplazamiento −0,572; escala 0,213). Cada jugador: gamma con ese desplazamiento y su forma y escala; si KS la rechaza, bootstrap de sus puntos.
- **Para la fase 4:** `data/interim/distribuciones_jugador.csv`; `dis.distribucion_vel(fila)` reconstruye la gamma de cada jugador.

### Simulación (fase 4)

- **Jugadores y modelo (D45):** los 30 jugadores con distribuciones (435 parejas). El ensamblaje se reentrena con los 24 partidos desde `data/processed/ensamblaje.joblib`, que guarda la 2.4.
- **Sorteo (D46):** independiente entre variables, jugadores y puntos; nivel y estatura fijos.
- **Partido (D47):** gana quien llega primero a 7; P(ganar) = Σ C(6 + k, k) p̄⁷ (1 − p̄)ᵏ, k = 0…6.
- **Ranking (D48):** todos contra todos (378 rivales por pareja); puntaje = probabilidad media de ganar el partido; réplicas hasta que el semiancho del intervalo t sea ≤ 0,005.
- **Incertidumbre (D49):** 100 réplicas de remuestreo de partidos; intervalos del puntaje y del puesto, % de réplicas en el puesto 1 y en el top 10.
- **Estatura y género (D50):** el modelo no cambia; la 4.5 compara la logística con estatura, con `mujer` en vez de estatura y sin ninguna, y muestra el ranking por categoría (dos hombres, mixta, dos mujeres).
- **Resultado:** `data/processed/ranking_parejas.csv`.

## Registro de decisiones

Ordenado como el notebook. Solo están las decisiones que aplican al notebook nuevo; la numeración se rehízo el 2026-11-23. El notebook anterior usa la numeración vieja, que queda en el historial de git. Toda decisión nueva se agrega aquí.

| # | Sección | Decisión | Fuente |
|---|---|---|---|
| D1 | General | Notebook de trabajo `notebooks/Proyecto_Padel.ipynb`, por fases; del notebook anterior se migra solo lo que sirve. Fase 1 en 5 pasos: entendimiento; faltantes, atípicos y tratamiento; variables preliminares; correlación; base unificada. Texto corto y concreto, como en los notebooks del profesor. | Equipo (2026-11-23) |
| D2 | General | Solo se registran conclusiones demostrables con código u observación directa de los datos; las interpretaciones se presentan al equipo como hipótesis. | Equipo (2026-11-07) |
| D3 | General | Ningún partido recibe tratamiento especial: solo reglas generales que el notebook calcula (duplicados D16, identidad D17, `y` = `x` → `y` vacía). | Equipo (2026-11-07) |
| D4 | General | `CLAUDE.md` solo tiene reglas de trabajo e hitos; las reglas de datos, las decisiones y los pendientes están en este README. | Equipo (2026-11-07) |
| D5 | 1.1 | Datos crudos en `data/raw/`, versionados en el repositorio privado: los dos Excel y el video en Parquet por partido (conversión exacta del CSV de 655 MB, que no se sube). Las salidas van a `data/interim/` y `data/processed/` y no se suben. | Equipo (2026-11-06 y 07) |
| D6 | 1.1 | Los datos de video se reciben ya generados; no se usa el pipeline de visión del profesor. | Revisión del repositorio del profesor (2026-11-06) |
| D7 | 1.1 | Variable objetivo: ganador del punto (`PUNTO_EQUIPO_1` en la traza; `gano` en la tabla del modelo). | Equipo (2026-11-06) |
| D8 | 1.1 / 1.2 | `TALLA` es el peso en kg; IMC = `TALLA` / (`ESTATURA` en m)². | Equipo (2026-11-06) |
| D9 | 1.2 | Jugadores sin formulario, con código entregado por el equipo: Alejandra Briget Deza Céspedes (407710), Juan Pablo Vanegas Herrera (356629) y Laura Sofía Rodríguez González (324764). Se imputan con datos de su mismo sexo: continuas con la media (más ruido N(0, sd) para las mujeres, con semilla fija y recorte al rango), ordinales con la mediana (la inferior si cae entre dos) y categóricas con la moda. Llevan `imputado = True`. Eliminarlos quitaría 77 de los 244 puntos completos. | Equipo (2026-11-06) |
| D10 | 1.2 | Códigos de 6 dígitos como texto. Jorge García: 99999 → `099999`. José Fabián García Camargo: 1075684546 → `107568` (primeros 6). | Equipo (2026-11-06) |
| D11 | 1.2 | Las variantes de un nombre son el mismo jugador; el nombre canónico es el completo, con dos apellidos cuando alguna fuente lo tiene. | Equipo (2026-11-06) |
| D12 | 1.2 / 1.3 | Escalas ordinales (0 = menor) de nivel, tiempo jugando, experiencia, estado físico y frecuencia de deporte. En frecuencia, «0 veces» y «No practico deportes» se unen en 0. `TIEMPO_JUGANDO_PADEL` se imputa como ordinal. | Equipo (2026-11-07) |
| D13 | 1.2 | `RIGTH` → `RIGHT`. Convención verificada con el video: BOTTOM en `y > 10` y RIGHT en `x > 5` para ambos equipos. | Verificado con los datos |
| D14 | 1.2 | Se descarta toda la pelota, incluida `player_hits_ball`: el 50,5 % de sus posiciones es relleno y el 10,1 % interpolación lineal. | Equipo (2026-11-07) |
| D15 | 1.2 / 1.3 | No se usa ninguna columna de pose ni de raqueta: faltan en el 3–25 % de los frames; `wrist_above_head` es verdadera en 21 frames y `racket_above_shoulder` en el 1,6 %; con el mismo jugador, la raqueta sobre el hombro baja en TOP para 19 de 20 y el giro de la raqueta sube para 14 de 20; el p90 del giro es 1.127 °/s con la raqueta corta en la imagen y 314 °/s con la larga. Por eso la limpieza solo trata posición y velocidad, y `body_scale` no se usa. | Equipo (2026-11-23) |
| D16 | 1.2 | Si un punto de video es copia exacta de otro del mismo partido, se conserva el de menor número (hoy, el 5-10 es copia del 5-9). Cada duplicado nuevo se revisa con el equipo. | Equipo (2026-11-06) |
| D17 | 1.2 | Identidad de cada track: por la etiqueta del video si en el partido ≥ 50 % de las filas tiene nombres de la alineación y ≥ 80 % concuerda con la casilla (fondo, lado) de la traza; si no, por posición contra la traza (asignación codiciosa por filas × p(fondo) × p(lado)). Hoy, los partidos 8, 17 y 34 van por posición. | Equipo (2026-11-06) |
| D18 | 1.2 | Los tracks genéricos (`Player_N`) y los que quedan sin casilla compatible se descartan. | Análisis |
| D19 | 1.2 | Se excluyen los puntos de video sin traza (7-11, 11-12, 23-9 y 32-13). La numeración de puntos del video coincide con la de la traza. | Análisis; numeración confirmada por el equipo (2026-11-07) |
| D20 | 1.2 | Limpieza de la posición y la velocidad: fps = 1/`duration` por punto; las posiciones hasta 0,5 m fuera de la cancha se recortan al borde y las más lejanas pasan a NaN; media móvil centrada de 0,5 s sobre `x` e `y`; velocidad y aceleración recalculadas sobre esa trayectoria; velocidad > 8 m/s o aceleración > 10 m/s² → NaN, interpolado si el hueco es de ≤ 5 frames (primero se quitan los saltos del tracking y luego se recalcula). | Equipo (2026-11-07) |
| D21 | 1.3 | 14 variables preliminares por jugador × punto en 3 grupos: nivel de juego (nivel: primera vez, principiante o intermedio/avanzado; tiempo jugando 0/1; experiencia; otra raqueta), tipo de juego (atacante, distancia media a la red, velocidad media, % quieto) y atributos físicos (estatura, IMC, edad, estado físico: malo/regular, bueno o excelente; frecuencia de deporte: 0, 1–2, 3–4 o más de 4 veces por semana; mujer). Por recomendación del profesor, las tres variables con más de dos categorías (nivel, estado físico y frecuencia) van en one-hot con `pd.get_dummies(drop_first=True)`: una columna 0/1 por categoría y la primera como base; las de dos categorías ya son 0/1. Las categorías de una sola persona se unen a la vecina (nivel avanzado → intermedio; estado físico malo → regular; tiempo jugando en 2 niveles). Si el track dura menos de 2 s, el jugador no tiene variables de video en ese punto. Motivo de los grupos: el profesor señaló que el problema de la fase 2 anterior estaba en cómo se agrupaban las variables. | Equipo (2026-11-23; one-hot por recomendación del profesor, 2026-11-24) |
| D22 | 1.3 | Las variables de video se miden respecto de su fondo: atacante = pasa al menos la mitad del punto más cerca de la red que la mediana de distancia de su fondo (7,66 m en BOTTOM y 6,53 m en TOP); quieto = por debajo de la mediana de velocidad de su fondo; a la distancia media a la red y a la velocidad media se les resta la mediana de su fondo. Motivo: el mismo jugador en TOP aparece 1,25 m más cerca de la red (17 de 18) y es atacante en más puntos (16 de 18), y el equipo 1 siempre está en BOTTOM. Tras medir así, la diferencia de distancia baja a −0,13 m. | Equipo (2026-11-23) |
| D23 | 1.3 | No se usan: la aceleración (su media en un punto es (velocidad final − velocidad inicial) / duración), las velocidades en píxeles (dependen de la distancia a la cámara), las columnas `_x`/`_y` y `body_scale` como variables, los puntos ganados (fuga) y las sumas, diferencias o razones entre jugadores. | Equipo (2026-11-23) |
| D24 | 1.4 | Dos variables «dicen lo mismo» si el valor absoluto de su r de Spearman es ≥ 0,8 por jugador × punto; se deja la del modelo 1 (D31). Salen `pct_quieto` (−0,83 con velocidad media) y `dist_red_m` (−0,81 con atacante): quedan 12 variables. | Equipo (2026-11-23) |
| D25 | 1.5 | Solo entran los puntos con los 4 jugadores identificados (244). | Equipo (2026-11-07) |
| D26 | 1.5 | Se eliminan los puntos en que algún jugador no tiene variables de video (9 del partido 17, sin `y`, y 11 con un track de menos de 2 s): quedan 224 puntos de 24 partidos y no se imputa nada de video. | Equipo (2026-11-23) |
| D27 | 1.5 | Del jugador al equipo: promedio de los dos jugadores (en las 0/1, la proporción del equipo). Dos filas por punto, una por equipo, con las variables propias (`propio_`) y las del rival (`rival_`) sin restarlas; objetivo `gano`. | Equipo (2026-11-23) |
| D28 | 2 | Fase 2 en 4 secciones con todos los temas del curso: 2.1 partición y validación; 2.2 naive y logística por pasos; 2.3 LASSO/Ridge, árbol, Random Forest, SVM, bagging, boosting y red neuronal; 2.4 ensamblaje y comparación final en test. | Equipo (2026-11-23) |
| D29 | 2.1 | Test 80/20 por partido; nunca se mezclan puntos de un mismo partido. Se ordenan los partidos por número (orden del torneo) y se toma 1 de cada 5. De los 5 arranques posibles se queda el que tiene ~20 % de los partidos, entre 18 % y 25 % de los puntos y la misma tasa de victorias del equipo 1 que el train (±5 puntos); si quedan varios, el de proporción de cancha 2 más parecida al total. Resultado: partidos 7, 14, 24, 29 y 39. El test se usa una sola vez, al final. | Equipo (2026-11-23; orden de los partidos confirmado por el equipo el 2026-11-24) |
| D30 | 2.1 | Evaluación común de todos los modelos: validación dejando un partido fuera en el train (19 pliegues, como `cross_val_score` del profesor pero por partido); las predicciones se simetrizan: p(A gana) = promedio de p(A contra B) y 1 − p(B contra A). Métrica principal: AUC (`roc_auc_score`), como en los notebooks del profesor; al lado, accuracy y F1. Intervalos del 95 % con bootstrap por partido de las predicciones de validación (1.000 remuestreos, semilla 2026), los mismos para todos los modelos. | Equipo (2026-11-24) |
| D31 | 1.4 / 2.2 | Selección por pasos con regresión logística (de `scikit-learn`, estandarizada dentro de cada pliegue): modelo 1 con `nivel_padel`, `atacante`, `vel_media_mps` y `estatura_cm` (orden fijado por lógica de juego antes de correr); luego el resto de nivel de juego y el de atributos físicos (tipo de juego no agrega variables tras D24). Un grupo entra solo si el intervalo del 95 % de la mejora de AUC queda por encima de 0 (parsimonia); si no, sus variables se prueban una por una, en orden, con la misma regla. | Equipo (2026-11-23; regla estricta y AUC 2026-11-24) |
| D32 | 2.3 | Cada modelo de la 2.3 repite la selección por pasos de D31 con sus propios hiperparámetros: empieza con las 4 variables del modelo 1 y cada uno termina con sus propias variables, escogidas entre las 12. | Equipo (2026-11-24) |
| D33 | 2.3 | Hiperparámetros con búsqueda bayesiana (`BayesSearchCV` de `scikit-optimize`), como en el notebook 05 del profesor: 40 evaluaciones por modelo y AUC como métrica, con un partido fuera por pliegue; se buscan una vez por modelo con las 4 variables del modelo 1 y se mantienen durante los pasos. Rangos: `C` de 0,0001 a 10; profundidad de 1 a 10; mínimo de 5 a 60 filas por hoja; de 25 a 500 árboles; tasa de aprendizaje de 0,001 a 1; `gamma` de 0,001 a 1; muestras de bagging del 30 % al 100 %; red de 4 a 64 neuronas y `alpha` de 0,001 a 10. Escoger y reportar con los mismos pliegues da un AUC algo optimista; el test de la 2.4 no tiene ese sesgo. Modelos: LASSO, Ridge, árbol, Random Forest, SVM, bagging, AdaBoost, Gradient Boosting, XGBoost y red neuronal (D35). | Equipo (2026-11-24) |
| D34 | 2.3 | Las probabilidades del SVM salen de `SVC(probability=True)`, como en el notebook del profesor; para calcularlas, `scikit-learn` hace una validación interna que mezcla puntos de un mismo partido. En scikit-learn 1.9 ese parámetro avisa que se quitará en la 1.11: `requirements.txt` pide `scikit-learn>=1.8,<1.11`. | Equipo (2026-11-24) |
| D35 | 2.3 | Red neuronal con `MLPClassifier` de `scikit-learn`, con la misma idea del notebook 08 del profesor: una capa densa de 4 a 64 neuronas (la búsqueda bayesiana no acepta listas de capas), activación relu o sigmoide, optimizador adam y lotes de 128. El profesor usa Keras/TensorFlow, pero TensorFlow no tiene versión estable para Python 3.14. Sin parada temprana (el profesor tampoco la usa), porque separaría una validación al azar que mezcla puntos de un mismo partido. | Equipo (2026-11-24) |
| D36 | 2.4 | Al ensamblaje entran los 4 mejores modelos de la 2.3 por AUC de validación. | Equipo (2026-11-24) |
| D37 | 2.4 | Voto ponderado igual que en la parte 5 del notebook 06 del profesor: cada modelo pesa α = (1 − error) / suma, con su accuracy de validación de la 2.3 (el profesor usa el error out-of-bag; aquí, el de la validación por partido; los dos son error en datos que el modelo no vio), y se suman las probabilidades en vez de los votos 0/1: p(ensamblaje) = Σ α × p(modelo). El resultado del ensamblaje en validación es algo optimista, porque los pesos salen de las mismas predicciones. | Equipo (2026-11-24) |
| D38 | 2.4 | Pasa a la fase 4 el de mayor AUC de validación entre el ensamblaje y los 10 modelos de la 2.3; se escoge antes de mirar el test. Resultado: el ensamblaje de red neuronal, Random Forest, SVM y Bagging. | Equipo (2026-11-24; ensamblaje confirmado por el equipo el mismo día) |
| D39 | 2.4 | Test: cada modelo se entrena con todo el train (sus hiperparámetros y sus variables) y se mide una sola vez en los 5 partidos de test, con AUC, accuracy y F1 e intervalo del 95 % por bootstrap de partidos (muy ancho con 5 partidos). | Equipo (2026-11-24) |
| D40 | 3 | Fase 3 en 3 secciones: 3.1 exploración; 3.2 ajuste con todos los jugadores; 3.3 ajuste por jugador y tabla para la fase 4. Solo se busca la distribución de `atacante` y `vel_media_mps`, las variables del ensamblaje (D38) que cambian de punto a punto; `nivel_padel` y `estatura_cm` son fijas del jugador y no se sortean. | Equipo (2026-11-24) |
| D41 | 3 | Datos: todas las filas jugador × punto con variables de video (979, de 30 jugadores), no solo las de los puntos del modelo, porque la distribución de un jugador no necesita a los otros 3 del punto. | Equipo (2026-11-24) |
| D42 | 3 | Cada jugador tiene su propia distribución; con menos de 10 puntos, usa la de todos los jugadores juntos. Motivo: chi² y Kruskal-Wallis rechazan que los jugadores sean iguales (p < 0,001). | Equipo (2026-11-24) |
| D43 | 3.2 | `atacante`: Bernoulli con p = proporción. `vel_media_mps`: 10 candidatas por máxima verosimilitud (normal, uniforme, triangular, exponencial, gamma, Erlang, chi², lognormal, Weibull y beta), con desplazamiento y escala estimados porque hay valores negativos; gana la de menor AIC entre las que KS no rechaza al 5 %; si todas se rechazan, bootstrap empírico. | Equipo (2026-11-24) |
| D44 | 3.3 | Por jugador: `atacante` con su proporción; `vel_media_mps` con la familia de la 3.2, el mismo desplazamiento para todos y la forma y la escala del jugador; si KS la rechaza en un jugador, bootstrap de sus puntos. Revisión: el percentil 90 de su distribución contra el intervalo del 95 % del percentil 90 de sus datos (bootstrap, 1.000 muestras, semilla 2026). | Equipo (2026-11-24) |
| D45 | 4 | Fase 4 en 5 secciones: 4.1 del punto al partido; 4.2 revisión con los partidos reales; 4.3 ranking de parejas; 4.4 incertidumbre del ranking; 4.5 parejas ideales según su estilo. Se simulan los 30 jugadores con distribuciones (435 parejas); los 2 que llenaron el formulario sin jugar no tienen video. El ensamblaje (D38) se reentrena con los 24 partidos (train y test) con los hiperparámetros de la 2.3 y los pesos de la 2.4, que la 2.4 guarda en `data/processed/ensamblaje.joblib`. | Equipo (2026-11-24) |
| D46 | 4 | Sorteo independiente: en cada punto simulado, cada jugador sortea `atacante` (Bernoulli con su p) y `vel_media_mps` (su gamma) por separado, entre variables, entre jugadores y entre puntos; nivel y estatura fijos; el equipo es el promedio de sus dos jugadores (D27). No conserva lo que mostró la 3.1: compañeros parecidos, rachas de un punto al siguiente y cambio entre tercios del torneo. | Equipo (2026-11-24) |
| D47 | 4.1 | El partido lo gana la primera pareja en llegar a 7 (así terminaron los 27 partidos, de 7-0 a 7-6). Con D46 todos los puntos de un enfrentamiento tienen la misma probabilidad p̄, así que P(ganar) = Σ C(6 + k, k) p̄⁷ (1 − p̄)ᵏ, k = 0…6; se comprueba con 10.000 partidos simulados punto por punto. | Equipo (2026-11-24) |
| D48 | 4.3 | Todos contra todos: cada pareja contra las 378 que no comparten jugador (82.215 enfrentamientos); puntaje = probabilidad media de ganar el partido. En cada enfrentamiento se sortean 4 puntos nuevos y se usa la fórmula de D47. Error de la simulación: réplicas independientes e intervalo t del 95 %; 5 réplicas piloto y, con su mayor semiancho, se calculan y completan las necesarias para que sea ≤ 0,005 (R = 5 × (semiancho / 0,005)²). | Equipo (2026-11-24; diseño de la incertidumbre delegado a Claude según lo usual en simulación) |
| D49 | 4.4 | Incertidumbre de los datos: 100 réplicas de remuestreo de partidos (semilla 2026); en cada una se reentrena el ensamblaje (mismos hiperparámetros y pesos), se reajustan las distribuciones por jugador (D42–D44) y se repite el todos contra todos con 1 sorteo por enfrentamiento. Por pareja: intervalo del 95 % (percentiles) del puntaje y del puesto, % de réplicas en el puesto 1 y en el top 10. | Equipo (2026-11-24; diseño delegado a Claude) |
| D50 | 4.5 | El puntaje sigue sobre todo a la estatura, y estatura y `mujer` tienen −0,78 (1.4). No se cambia el modelo (sus variables se fijaron antes de correr, D31); la 4.5 agrega (a) la logística de la 2.2 con `mujer` en vez de estatura y sin ninguna de las dos, con la misma validación (D30), y (b) el ranking por categoría (dos hombres, mixta, dos mujeres). La limitación se reporta: con estos datos no se puede separar la estatura del género. | Equipo (2026-11-24, por recomendación de Claude) |

## Supuestos y pendientes

| Tema | Detalle | Acción |
|---|---|---|
| Datos faltantes | 15 puntos de la traza sin video. Los 12 partidos programados sin traza ni video aparecen en `Partidos_Jugados` con 0 puntos. | Confirmar si se jugaron; si llegan datos, se vuelve a correr la fase 1 |

## Origen

El proyecto parte del material del profesor (`padel-vision-analytics-light`). De ese material solo se conservó lo necesario para la analítica: la documentación de las columnas del CSV de video (reescrita en `docs/diccionario.md`), el mapa de keypoints y las dimensiones de la cancha. No se usa el pipeline de visión (YOLO, pose, tracking), porque los CSV de video se reciben ya generados.
