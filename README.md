# Proyecto final de analítica: parejas ideales de pádel

Encontrar las parejas que maximizan la probabilidad de ganar el torneo según su estilo de juego, a partir de datos de video (pose y posición por frame), la traza de partidos y el formulario de jugadores.

El contexto, las reglas de trabajo y el registro de decisiones están en [CLAUDE.md](CLAUDE.md). La guía completa está en [docs/guia_contexto_proyecto.pdf](docs/guia_contexto_proyecto.pdf).

## Fases

1. **Manejo de datos** (actual): limpieza, imputación, granularidad y variables de pareja.
2. **Predictiva:** regresión logística regularizada y Random Forest sobre el ganador del punto.
3. **Distribuciones:** distribución empírica por jugador y variable aleatoria.
4. **Simulación:** combinaciones de parejas evaluadas con el modelo de la fase 2.

## Estructura

```
CLAUDE.md                  Contexto, reglas y decisiones del proyecto
docs/
  guia_contexto_proyecto.pdf
  diccionario_datos.md     Columnas de las tres fuentes y uso de cada una
  assets/                  Mapa de keypoints de pose
src/                       Código reutilizable del pipeline
notebooks/                 Exploración y reportes
data/
  raw/                     Datos de entrada sin modificar (no versionados)
  interim/                 Datos limpios intermedios (no versionados)
  processed/               Tablas finales (no versionadas)
referencia/                Script del profesor para leer la traza (solo consulta)
```

## Datos

Los datos no se suben al repositorio por su tamaño (un CSV por punto) y porque el formulario tiene datos personales. Cada integrante los copia en `data/raw/`:

- `Partido_X_Punto_Y_CON_NOMBRES.csv`: CSV de video, uno por punto.
- `CANCHA2.csv` (y `CANCHA1.csv` si existe): traza de partidos.
- Formulario de jugadores.

## Instalación

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
source .venv/bin/activate     # macOS / Linux
pip install -r requirements.txt
```

## Origen

Este repositorio parte del material del profesor (`padel-vision-analytics-light`). De ese material solo se conservó lo necesario para la analítica: la documentación de las columnas del CSV (reescrita en `docs/diccionario_datos.md`), el mapa de keypoints, las dimensiones de la cancha y el script de lectura de la traza. Se excluyeron el pipeline de visión (YOLO, pose, tracking, notebooks, anotaciones de cancha) porque los CSV se reciben ya generados, y la base normalizada porque corresponde a otro torneo.
