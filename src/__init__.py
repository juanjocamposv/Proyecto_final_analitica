"""Funciones reutilizables del pipeline de analítica de pádel (parejas ideales).

Módulos:
- datos: rutas, constantes de la cancha, lectura de las fuentes y utilidades de reporte.
- identidades: nombres canónicos, alias y códigos de estudiante.
- fuentes: limpieza de la traza y del formulario.
- video: identidad de cada track, faltantes, evidencia de la pelota y limpieza de posición y velocidad.
- variables: variables jugador × punto, correlación y tabla equipo × punto para el modelo.
- modelo: fase 2 (partición, evaluación por partido, selección por pasos, búsqueda bayesiana, ensamblaje y test).
- distribuciones: fase 3 (exploración, ajuste de distribuciones con todos los jugadores y por jugador).
- simulacion: fase 4 (del punto al partido, todos contra todos y remuestreo de partidos).
"""
