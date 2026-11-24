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
| Código y resultados | [notebooks/Proyecto_Padel.ipynb](notebooks/Proyecto_Padel.ipynb) y `src/` (el notebook anterior, `Proyecto_Parejas_Padel.ipynb`, solo se consulta para migrar la fase 3) |

## Hitos

Notebook nuevo (D1): se migra por fase lo que sirve del anterior.

- [x] **Fase 1** (1.1–1.5): entendimiento; faltantes, atípicos y tratamiento; 14 variables preliminares (D21, D22, D15); correlación (D24, quedan 12); base unificada → `data/processed/tabla_modelo.csv` (224 puntos, 448 filas, D26).
- [x] **Fase 2** (2.1–2.4): test 80/20 por partido y evaluación común con AUC (D28–D30), logística por pasos con regla estricta (D31), diez modelos con búsqueda bayesiana por partido y pasos por modelo (D32–D35), voto ponderado por accuracy con los 4 mejores (D36–D37), modelo final por AUC de validación (D38) y test con todos los modelos (D39).
- [ ] **Fase 3** Migrar las distribuciones con las variables nuevas.
- [ ] **Fase 4** Simulación de parejas y ranking con incertidumbre.

## Cómo trabajar

1. **Preguntar antes de decidir.** Nada de datos, métodos, umbrales o estructura se asume. Si algo no está en el README, se pregunta con opciones concretas y una recomendada, con la evidencia que haga falta para decidir. Si un dato falta o un supuesto no está confirmado, se dice. Antes de eliminar una parte o una conclusión, se confirma con el equipo.
2. **Solo conclusiones demostrables (D2).**
   - En el notebook, el README y el diccionario solo se escribe lo que el código o la observación directa de los datos demuestran.
   - Las interpretaciones («la homografía está desplazada», «parece remuestreado») no se registran; si importan, se le presentan al equipo como hipótesis.
   - Ningún partido recibe tratamiento especial: solo reglas generales (D3).
3. **Registrar sin llenar este archivo.**
   - Cada decisión nueva va al registro del README y se cita con su `D#` en la sección del notebook.
   - Las reglas de datos nuevas van a «Reglas de datos» del README.
   - Aquí solo se actualizan los hitos y, si cambian, las reglas de trabajo.
4. **Notebook único (D1).** Todo avance se agrega en su sección de `Proyecto_Padel.ipynb`, con el formato Objetivo → Decisiones (D#) → Código → Resultado. El texto es corto y concreto, como en los notebooks del profesor: sin relleno, sin repetir lo ya dicho en otra sección. No se crean más notebooks.
5. **Código en `src/`.** Son 7 módulos: `datos`, `identidades`, `fuentes`, `video`, `variables`, `modelo` (fase 2) y `distribuciones` (fase 3); la fase 4 agrega `simulacion.py`.
   - Funciones reutilizables para cualquier `(partido, punto)`.
   - Los casos especiales se detectan con reglas generales, no con listas de partidos.
6. **Verificar antes de reportar.**
   - Ejecutar el notebook completo (por ejemplo con `nbclient`; hoy tarda unos 30 min) sin errores.
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
