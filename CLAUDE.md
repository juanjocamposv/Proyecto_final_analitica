# CLAUDE.md — Pádel: parejas ideales

Responde en español, de forma directa y técnica.

**Objetivo del proyecto:** encontrar las parejas de pádel que maximizan la probabilidad de ganar el torneo según su estilo de juego.

## Dónde está cada cosa

Este archivo solo tiene reglas de trabajo e hitos. Todo lo concreto está en otros archivos y se lee ahí cuando hace falta:

| Necesito… | Está en |
|---|---|
| Reglas de datos (fuentes, calidad por partido, identidad, limpieza, variables, modelado) | [README.md § Reglas de datos](README.md#reglas-de-datos) |
| Decisiones tomadas (D#) | [README.md § Registro de decisiones](README.md#registro-de-decisiones) |
| Supuestos sin confirmar y pendientes | [README.md § Supuestos y pendientes](README.md#supuestos-y-pendientes) |
| Significado, unidad y uso de cada columna | [docs/diccionario.md](docs/diccionario.md) (A: fuentes crudas; B: tablas derivadas) |
| Qué hace cada archivo | [README.md § Archivos del repositorio](README.md#archivos-del-repositorio) |
| Código y resultados | [notebooks/Proyecto_Parejas_Padel.ipynb](notebooks/Proyecto_Parejas_Padel.ipynb) y `src/` |

## Hitos

- [x] **1.1–1.7** Fuentes, entendimiento (1.2.1–1.2.9, incluye pelota y atípicos), identidad de jugadores, traza, formulario, calidad e identidad del video.
- [x] **1.8** Limpieza del video frame × jugador → `data/interim/video_limpio.parquet`.
- [x] **1.9** Variables jugador × punto (estilo con la línea de saque, ritmo, profundidad, movimiento, raqueta/pose, golpes) → `participaciones` y `jugadores`.
- [x] **1.10** Variables pareja × punto y filtros sin resultado → `data/processed/tabla_modelo.csv` (244 puntos, 161 variables).
- [x] **1.11** Variables para las fases 2 a 4 (D34): 19 `dif_` reconstruibles desde cada jugador.
- [x] **Cierre de la fase 1** Commit y push (`9847c3c`).
- [x] **2.1–2.4** Partición por partido (D35), evaluación común `src/modelo.py` (D36), logística (D37), LASSO/Ridge y árbol (D38).
- [x] **2.5–2.6** Random Forest y SVM con tuning (D39–D41), bagging y boosting y red neuronal (D42–D45).
- [x] **2.7** Ensamblaje de Gradient Boosting, LASSO y RF: promedio, voto ponderado y voto mayoritario (D46).
- [x] **2.7–2.8** Mejora del ensamblaje (D47), regla fija y comparación en test con bootstrap (D48), modelo para la fase 4 (D49). En test ningún modelo supera al naive: decidir con el equipo cómo sigue la fase 4.
- [x] **Fase 2 (detalle)** Se replica al pie de la letra la forma de los notebooks del profesor (`notebooks/referencias/`). Reparto: 2.1–2.4 (preparación, evaluación común, naive y logística, Lasso/Ridge y árbol) y 2.5–2.8 (Random Forest y SVM con tuning, bagging y boosting, ensamblaje, bootstrap y comparación final).
- [x] **Fase 3** Distribuciones por jugador y variable (D50–D57): exploración 3.1–3.3, ajuste global 3.4, por jugador y bootstrap 3.5, tabla final 3.6 → `distribuciones_global.csv` y `distribuciones_jugador.csv`. Quedan 3 decisiones para la fase 4 (cómo sortear, dependencia entre puntos y cambio de nivel).
- [ ] **Fase 4** Simulación de parejas y ranking con incertidumbre.

## Cómo trabajar

1. **Preguntar antes de decidir.** Nada de datos, métodos, umbrales o estructura se asume. Si algo no está en el README, se pregunta con opciones concretas y una recomendada, con la evidencia que haga falta para decidir. Si un dato falta o un supuesto no está confirmado, se dice. Antes de eliminar una parte o una conclusión, se confirma con el equipo.
2. **Solo conclusiones demostrables (D30).**
   - En el notebook, el README y el diccionario solo se escribe lo que el código o la observación directa de los datos demuestran.
   - Las interpretaciones («la homografía está desplazada», «parece remuestreado») no se registran; si importan, se le presentan al equipo como hipótesis.
   - Ningún partido recibe tratamiento especial: solo reglas generales (D26).
3. **Registrar sin llenar este archivo.**
   - Cada decisión nueva va al registro del README y se cita con su `D#` en la sección del notebook.
   - Las reglas de datos nuevas van a «Reglas de datos» del README.
   - Aquí solo se actualizan los hitos y, si cambian, las reglas de trabajo.
4. **Notebook único.** Todo avance se agrega en su sección de `Proyecto_Parejas_Padel.ipynb`, con el formato Objetivo → Decisiones (D#) → Código → Resultado. No se crean notebooks nuevos.
5. **Código en `src/`.** Son 7 módulos: `datos`, `identidades`, `fuentes`, `video`, `variables`, `modelo` (fase 2) y `distribuciones` (fase 3); la fase 4 agrega `simulacion.py`.
   - Funciones reutilizables para cualquier `(partido, punto)`.
   - Los casos especiales se detectan con reglas generales, no con listas de partidos.
6. **Verificar antes de reportar.**
   - Ejecutar el notebook completo (por ejemplo con `nbclient`, ~20 min por las búsquedas de la fase 2) sin errores.
   - No reportar números sin haberlos corrido.
   - Revisar los gráficos generados.
7. **Datos.**
   - No modificar `data/raw/`; los resultados se escriben en archivos nuevos con `datos.guardar_tabla`.
   - Mostrar un resumen antes y después de cada paso (`datos.resumen_tabla`, `datos.comparar`).
   - Documentar cada tabla derivada en la parte B del diccionario.
   - Semilla: `datos.SEMILLA` = 2026.
8. **Gráficos (matplotlib).** Una serie en `#2a78d6`, sin leyenda si hay una sola serie, cuadrícula tenue y bordes superior y derecho ocultos.
9. **Repositorio simple.** Antes de crear un archivo, ver si cabe en uno existente. Si cambia la estructura, actualizar la tabla de archivos del README.
10. **Git.** El repositorio es privado y los datos están versionados (Excel y video en Parquet). Commit y push solo cuando la fase 1 esté terminada y completa (y luego, solo cuando el equipo lo pida). Antes de hacer push, hacer `fetch` y combinar los cambios del equipo.
11. **Cierre de sesión.** Resumir en viñetas qué se decidió, qué archivos se generaron y qué queda pendiente.
