# CLAUDE.md — Pádel: parejas ideales

Responde en español, de forma directa y técnica.

## Objetivo

Encontrar las parejas de pádel que maximizan la probabilidad de ganar el torneo según su estilo de juego.

## Fases

1. **Manejo de datos (fase actual):** limpieza, imputación, granularidad y variables artificiales (diferencias, razones, sumas, combinaciones).
2. **Predictiva:** modelo supervisado que predice el ganador del punto.
3. **Distribuciones:** una distribución empírica por variable aleatoria y por jugador. Los valores fijos (altura, peso, nivel) no se modelan como distribución; entran como atributos.
4. **Simulación:** combinaciones de parejas y partidos muestreando las distribuciones de la fase 3, evaluadas con el modelo de la fase 2.

## Fuentes de datos

| Fuente | Granularidad | Contenido |
|---|---|---|
| CSV de video `Partido_X_Punto_Y` | frame × jugador | fps variable por video (`duration` = 1/fps), 67 columnas. Posición, movimiento, pose, velocidad de raqueta. Solo existe el partido 24, punto 1 (1.022 frames, 4 jugadores, 4.068 filas). |
| Traza de partidos (cancha 2) | punto | 11 partidos, 119 puntos, partidos a 7 puntos. Ganador (`PUNTO_EQUIPO_1`), marcador, parejas, LEFT/RIGHT, TOP/BOTTOM. |
| Formulario de jugadores | jugador | 29 filas. Datos fijos. |

- **Llave de unión:** `(ID_PARTIDO, ID_PUNTO, jugador)` con `CODIGO_ESTUDIANTE`. Nunca unir por nombre: no coincide entre archivos.
- Partido 24, punto 1: Juliana Gómez + Santiago Pujol (bottom) vs. Andrés Santafé + Laura Rodríguez (top). Ganó el equipo 2.
- **Detalle de columnas** (unidades, cómo se generan y uso de cada una): [docs/diccionario_datos.md](docs/diccionario_datos.md). Consultarlo antes de usar una columna.
- Coordenadas del video en metros: `x` ∈ [0, 10], `y` ∈ [0, 20], red en `y = 10`, equipo `top` en `y < 10`. Constantes en `src/constantes_cancha.py`.

## Repositorio

- `data/raw/`: datos de entrada sin modificar. `data/interim/` y `data/processed/`: salidas. Ninguna carpeta de datos se versiona.
- `src/`: funciones reutilizables del pipeline. `notebooks/`: exploración y reportes que importan de `src/`.
- `referencia/`: script del profesor para leer la traza; solo consulta, no se modifica ni se importa.
- No se usa el pipeline de visión del profesor (YOLO, pose, tracking): los CSV de video se reciben ya generados.

## Reglas de limpieza del video

- **Pelota:** descartar todas sus columnas (posiciones de relleno, `ball_speed_mps` = 0, `player_hits_ball` siempre False). Aproximar golpes con picos de velocidad de raqueta.
- **Píxeles:** dividir toda medida en píxeles por `body_scale` (va de ~23 a ~71 según la distancia a cámara). Los ángulos y las variables `*_norm` ya son comparables.
- **Velocidad y aceleración:** las posiciones están cuantizadas (~5 cm); no usar las columnas originales. Suavizar la trayectoria y recalcular con ventanas de 0,5 a 1 s.
- **Pose:** interpolar solo huecos de ≤5 frames; el resto queda NaN. Guardar `pose_coverage` por jugador y punto (falta sistemáticamente: Andrés 27 %).
- **Faltantes menores:** `team` se completa con el valor constante del jugador; distancia al compañero se recalcula desde las posiciones.
- **Rol:** derivarlo de la posición; `zone` no es un rol real.
- **Descartadas:** `duration` (solo para calcular fps), `total_frames`, `video_duration`, `prev_*`, coordenadas en píxeles (solo para derivar), `upper_arm_length`, `forearm_length`, `arm_extension`, `racket_vector_x/y`, orientaciones de antebrazo y brazo.

## Reglas del formulario y la traza

- **Fuga de datos:** `Puntos_Total`, `Partidos_Jugados` y `Puntos_Por_Partido` nunca son predictoras; solo sirven para validar.
- **Descartadas:** ciudad, programa, semestre, razón y objetivo de participación, interés y clases de pádel, desayuno, género musical, comentarios, `RITMO_CARDIACO`.
- **Usables:** `ESTATURA`, `TALLA`, `MANO_DOMINANTE` (solo 2 zurdos), `NIVEL_ACTUAL_PADEL` (ordinal), `TIEMPO_JUGANDO_PADEL`, `EXPERIENCIA_PADEL`, `PRACTICA_OTRO_DEPORTE_RAQUETA`, `ESTADO_FISICO`, `FRECUENCIA_DEPORTE`, `EDAD`, `GENERO`, `LESIONES_LY`.
- **Lado de cancha:** el equipo 1 siempre está en BOTTOM. Incluirlo como variable de control, sin sacar conclusiones fuertes.
- **Nombres:** usar la tabla de equivalencias. Separar coincidencias seguras de dudosas; no asignar códigos a las dudosas sin confirmación.
- **Búsqueda en Teams (proceso recurrente):** cada vez que un jugador no cruza con el formulario (ahora o cuando lleguen nuevos videos o la traza de la cancha 1), listarlo y pedir al equipo que lo busque manualmente en Teams. Con el nombre y código confirmados, actualizar la tabla de equivalencias y volver a correr la imputación. Se repite hasta que no queden jugadores sin cruzar.

## Variables

Granularidad: **frame × jugador → jugador × punto → pareja × punto** (con la pareja rival como contexto).

- **Jugador × punto:** profundidad media y desviación, % cerca de la red, lado, área cubierta; distancia recorrida, % quieto, ráfagas rápidas, velocidad p90; % raqueta sobre el hombro, proporción por dirección de raqueta, variabilidad de ángulos; velocidad de raqueta normalizada en p90/p95 (no el máximo) y conteo de picos; `pose_coverage`.
- **Pareja × punto:** diferencias (complementariedad), razones min/max (balance), sumas (capacidad total), cohesión (media y desviación de `distance_player_to_teammate_m`), sincronía (correlación de movimiento entre compañeros), fijas de pareja (estatura, nivel, experiencia, género, manos y si cada jugador está en el lado de su mano), relativas al rival (A − B), contexto (lado, cobertura de pose).
- **Meta:** 10 a 15 variables finales. Revisar correlación, redundancia, varianza baja y faltantes antes de elegir.

## Modelado

- **Objetivo:** `PUNTO_EQUIPO_1` por punto. La probabilidad del partido a 7 puntos se deriva de p(ganar punto).
- **Validación:** dejar un partido fuera (los puntos de un partido no son independientes). Nunca mezclar puntos del mismo partido entre entrenamiento y prueba.
- **Modelos:** regresión logística regularizada (referencia) vs. Random Forest.
- **Métricas:** AUC, log-loss, calibración e importancia de variables.
- **Distribuciones (fase 3):** comparar bootstrap empírico contra paramétrica ajustada, considerando las pocas observaciones por jugador.
- **Simulación (fase 4):** ranking de parejas por probabilidad media de ganar contra el resto, con intervalos de incertidumbre.
- Parejas casi irrepetibles (~22 parejas, 25 jugadores): el modelo aprende de características de pareja, no de identidades.

## Prácticas de código

- Python con pandas, numpy y scikit-learn.
- Ejecutar el código y verificar resultados; no reportar números sin correrlo.
- Pipeline como funciones reutilizables para cualquier `Partido_X_Punto_Y`, no código específico del partido 24.
- Mostrar resumen antes y después de cada paso de limpieza (filas, faltantes, rangos).
- No modificar los datos crudos; escribir los resultados en archivos nuevos.
- Entregar cada tabla derivada con su diccionario de variables.

## Registro de decisiones (fase 1)

| # | Decisión | Fuente |
|---|---|---|
| D1 | `TALLA` = peso en kg. Se calcula IMC = `TALLA` / (`ESTATURA` en m)². | Confirmado por el equipo (2026-11-06) |
| D2 | `racket_speed`, `wrist_speed`, `elbow_speed` en px/s; velocidades angulares en °/s. Los fps se obtienen por CSV como 1/`duration`, no se asume 60. | Código del pipeline del profesor (ver `docs/diccionario_datos.md`) |
| D3 | Variable objetivo: ganador del punto (`PUNTO_EQUIPO_1`). | Confirmado por el equipo (2026-11-06) |
| D4 | Imputación de los 3 jugadores sin datos en el formulario (Alejandra Briget Deza Céspedes, Juan Pablo Vanegas Herrera, Laura Sofía Rodríguez González): el equipo entrega el código de estudiante de cada uno. Numéricas continuas (estatura, peso, edad, tiempo jugando, etc.): media de su sexo; a las dos mujeres se les suma ruido N(0, sd de las mujeres) con semilla fija y recorte al rango observado. Ordinales (nivel, experiencia, estado físico, frecuencia): mediana de su sexo. Categóricas (mano, lesiones, otro deporte de raqueta): moda de su sexo. Marcar estas filas con `imputado = True`. | Decisión del equipo (2026-11-06) |
| D5 | Fila sin nombre (152 cm, 48 kg): el equipo entrega el nombre encontrado en Teams y se usa esa fila para completar la información de ese jugador. | Decisión del equipo (2026-11-06) |
| D6 | Los CSV de video se reciben ya generados; del material del profesor solo se conserva lo necesario para la analítica. Su base normalizada (`data/normalized/`) no se usa: es de otro torneo (su partido 24 tiene otros jugadores). | Revisión del repositorio del profesor (2026-11-06) |

**Cuándo se necesita la información de D4 y D5:** en el paso de tabla de equivalencias de nombres, después del pipeline de limpieza del video y antes de construir las variables por jugador y por pareja. Pedirla al equipo al empezar ese paso.

## Supuestos sin confirmar

Si un dato falta o un supuesto no está confirmado, decirlo y preguntar antes de asumirlo.

- Faltan los CSV de video de 118 de 119 puntos y posiblemente la traza de la cancha 1 (partidos impares).

## Cierre de sesión

Al terminar, resumir en viñetas: qué se decidió, qué archivos se generaron y qué queda pendiente.
