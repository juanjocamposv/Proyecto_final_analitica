"""Dimensiones de la cancha de pádel en el sistema de coordenadas del CSV de video.

x en [0, ANCHO_M] a lo ancho, y en [0, LARGO_M] a lo largo.
(0, 0) es la esquina del fondo lejano izquierdo; el equipo `top` está en y < Y_RED_M.
"""

ANCHO_M = 10.0
LARGO_M = 20.0
Y_RED_M = LARGO_M / 2

# Distancia de la red a la línea de saque (límite de los cuadros de servicio).
DIST_RED_LINEA_SAQUE_M = 6.95
