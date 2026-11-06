"""
Script para normalizar las bases de datos CANCHA1.csv y CANCHA2.csv
Genera tablas relacionales con Foreign Keys (FK) apropiadas

Estructura generada:
1. jugadores.csv - Catálogo de jugadores únicos
2. partidos.csv - Información de partidos
3. puntos.csv - Puntos jugados en cada partido
4. participaciones_jugadores.csv - Relación jugador-partido-punto-posición (con FK única)

Autor: Sistema de normalización de base de datos Padel
"""

import pandas as pd
import os
from pathlib import Path

def normalize_padel_database(cancha1_path, cancha2_path, output_dir="data/normalized"):
    """
    Normaliza las bases de datos de CANCHA1 y CANCHA2 en tablas relacionales
    
    Args:
        cancha1_path: Ruta al archivo CANCHA1.csv
        cancha2_path: Ruta al archivo CANCHA2.csv
        output_dir: Directorio donde se guardarán los CSVs normalizados
    """
    
    # Crear directorio de salida si no existe
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    print("Current working directory:", os.getcwd())
    
    print("🎾 Iniciando normalización de base de datos de Padel...")
    print("="*70)
    
    # Leer archivos originales
    print(f"\n📂 Leyendo archivos...")
    df_cancha1 = pd.read_csv(os.path.join(os.getcwd(), cancha1_path), encoding="latin-1", sep=";")
    df_cancha2 = pd.read_csv(os.path.join(os.getcwd(), cancha2_path), encoding="latin-1", sep=";")
    
    print(f"   ✓ CANCHA1: {len(df_cancha1)} registros")
    print(f"   ✓ CANCHA2: {len(df_cancha2)} registros")
    
    # Combinar ambas canchas
    df_combined = pd.concat([df_cancha1, df_cancha2], ignore_index=True)
    print(f"   ✓ Total combinado: {len(df_combined)} registros")
    
    # ==========================================
    # 1. TABLA: jugadores.csv
    # ==========================================
    print(f"\n📊 Generando tabla: jugadores.csv")
    
    # Extraer todos los nombres únicos de jugadores
    jugadores_set = set()
    
    for _, row in df_combined.iterrows():
        # Agregar solo si no es NaN
        if pd.notna(row['ID_JUGADOR_1_EQUIPO_1']):
            jugadores_set.add(row['ID_JUGADOR_1_EQUIPO_1'])
        if pd.notna(row['ID_JUGADOR_2_EQUIPO_1']):
            jugadores_set.add(row['ID_JUGADOR_2_EQUIPO_1'])
        if pd.notna(row['ID_JUGADOR_1_EQUIPO_2']):
            jugadores_set.add(row['ID_JUGADOR_1_EQUIPO_2'])
        if pd.notna(row['ID_JUGADOR_2_EQUIPO_2']):
            jugadores_set.add(row['ID_JUGADOR_2_EQUIPO_2'])
    
    # Crear DataFrame de jugadores con ID único
    jugadores_list = sorted(list(jugadores_set))
    df_jugadores = pd.DataFrame({
        'id_jugador': range(1, len(jugadores_list) + 1),
        'nombre_completo': jugadores_list
    })
    
    # Crear diccionario para mapeo rápido nombre -> id
    nombre_to_id = dict(zip(df_jugadores['nombre_completo'], df_jugadores['id_jugador']))
    
    print(f"   ✓ {len(df_jugadores)} jugadores únicos identificados")
    
    # Guardar
    jugadores_path = os.path.join(output_dir, 'jugadores.csv')
    df_jugadores.to_csv(jugadores_path, index=False, encoding='utf-8-sig')
    print(f"   ✓ Guardado en: {jugadores_path}")
    
    # ==========================================
    # 2. TABLA: partidos.csv
    # ==========================================
    print(f"\n📊 Generando tabla: partidos.csv")
    
    # Extraer partidos únicos
    df_partidos = df_combined[['ID_PARTIDO', 'CANCHA']].drop_duplicates().sort_values('ID_PARTIDO').reset_index(drop=True)
    df_partidos.columns = ['id_partido', 'cancha']
    
    print(f"   ✓ {len(df_partidos)} partidos únicos")
    
    # Guardar
    partidos_path = os.path.join(output_dir, 'partidos.csv')
    df_partidos.to_csv(partidos_path, index=False, encoding='utf-8-sig')
    print(f"   ✓ Guardado en: {partidos_path}")
    
    # ==========================================
    # 3. TABLA: puntos.csv
    # ==========================================
    print(f"\n📊 Generando tabla: puntos.csv")
    
    # Extraer puntos únicos (id_partido + numero_punto)
    df_puntos_temp = df_combined[['ID_PARTIDO', 'ID_PUNTO']].drop_duplicates()
    df_puntos_temp = df_puntos_temp.sort_values(['ID_PARTIDO', 'ID_PUNTO']).reset_index(drop=True)
    
    # Asignar ID único a cada punto
    df_puntos = pd.DataFrame({
        'id_punto': range(1, len(df_puntos_temp) + 1),
        'id_partido': df_puntos_temp['ID_PARTIDO'].values,
        'numero_punto': df_puntos_temp['ID_PUNTO'].values
    })
    
    # Crear diccionario para mapeo rápido (id_partido, numero_punto) -> id_punto
    punto_key_to_id = {}
    for _, row in df_puntos.iterrows():
        key = (row['id_partido'], row['numero_punto'])
        punto_key_to_id[key] = row['id_punto']
    
    print(f"   ✓ {len(df_puntos)} puntos únicos")
    
    # Guardar
    puntos_path = os.path.join(output_dir, 'puntos.csv')
    df_puntos.to_csv(puntos_path, index=False, encoding='utf-8-sig')
    print(f"   ✓ Guardado en: {puntos_path}")
    
    # ==========================================
    # 4. TABLA: participaciones_jugadores.csv (LA MÁS IMPORTANTE!)
    # ==========================================
    print(f"\n📊 Generando tabla: participaciones_jugadores.csv")
    print(f"   Esta tabla contiene la FK única para cada jugador en cada contexto")
    
    participaciones = []
    id_participacion = 1
    
    for _, row in df_combined.iterrows():
        id_partido = row['ID_PARTIDO']
        numero_punto = row['ID_PUNTO']
        
        # Saltar registros con datos faltantes
        if pd.isna(id_partido) or pd.isna(numero_punto):
            continue
        
        # Obtener id_punto de la tabla puntos
        punto_key = (id_partido, numero_punto)
        if punto_key not in punto_key_to_id:
            continue
            
        id_punto = punto_key_to_id[punto_key]
        
        # Jugador 1 Equipo 1
        if pd.notna(row['ID_JUGADOR_1_EQUIPO_1']) and row['ID_JUGADOR_1_EQUIPO_1'] in nombre_to_id:
            participaciones.append({
                'id_participacion': id_participacion,
                'id_jugador': nombre_to_id[row['ID_JUGADOR_1_EQUIPO_1']],
                'id_punto': id_punto,
                'equipo': 1,
                'posicion_equipo': row['EQUIPO_1_POSICION'].strip(),
                'posicion': row['POSICION_JUGADOR_1_EQUIPO_1'].strip(),
                'punto_ganado': 1 if row['PUNTO_EQUIPO_1'] == 1 else 0
            })
            id_participacion += 1
        
        # Jugador 2 Equipo 1
        if pd.notna(row['ID_JUGADOR_2_EQUIPO_1']) and row['ID_JUGADOR_2_EQUIPO_1'] in nombre_to_id:
            participaciones.append({
                'id_participacion': id_participacion,
                'id_jugador': nombre_to_id[row['ID_JUGADOR_2_EQUIPO_1']],
                'id_punto': id_punto,
                'equipo': 1,
                'posicion_equipo': row['EQUIPO_1_POSICION'].strip(),
                'posicion': row['POSICION_JUGADOR_2_EQUIPO_1'].strip(),
                'punto_ganado': 1 if row['PUNTO_EQUIPO_1'] == 1 else 0
            })
            id_participacion += 1
        
        # Jugador 1 Equipo 2
        if pd.notna(row['ID_JUGADOR_1_EQUIPO_2']) and row['ID_JUGADOR_1_EQUIPO_2'] in nombre_to_id:
            participaciones.append({
                'id_participacion': id_participacion,
                'id_jugador': nombre_to_id[row['ID_JUGADOR_1_EQUIPO_2']],
                'id_punto': id_punto,
                'equipo': 2,
                'posicion_equipo': row['EQUIPO_2_POSICION'].strip(),
                'posicion': row['POSICION_JUGADOR_1_EQUIPO_2'].strip(),
                'punto_ganado': 1 if row['PUNTO_EQUIPO_2'] == 1 else 0
            })
            id_participacion += 1
        
        # Jugador 2 Equipo 2
        if pd.notna(row['ID_JUGADOR_2_EQUIPO_2']) and row['ID_JUGADOR_2_EQUIPO_2'] in nombre_to_id:
            participaciones.append({
                'id_participacion': id_participacion,
                'id_jugador': nombre_to_id[row['ID_JUGADOR_2_EQUIPO_2']],
                'id_punto': id_punto,
                'equipo': 2,
                'posicion_equipo': row['EQUIPO_2_POSICION'].strip(),
                'posicion': row['POSICION_JUGADOR_2_EQUIPO_2'].strip(),
                'punto_ganado': 1 if row['PUNTO_EQUIPO_2'] == 1 else 0
            })
            id_participacion += 1
    
    df_participaciones = pd.DataFrame(participaciones)
    
    print(f"   ✓ {len(df_participaciones)} participaciones generadas")
    print(f"   ✓ Cada registro tiene una FK única: (id_jugador, id_punto, equipo, posicion)")
    
    # Guardar
    participaciones_path = os.path.join(output_dir, 'participaciones_jugadores.csv')
    df_participaciones.to_csv(participaciones_path, index=False, encoding='utf-8-sig')
    print(f"   ✓ Guardado en: {participaciones_path}")
    
    # ==========================================
    # 5. TABLA AUXILIAR: jugador_partido_posicion.csv
    # ==========================================
    print(f"\n📊 Generando tabla auxiliar: jugador_partido_posicion.csv")
    print(f"   Esta tabla facilita el mapeo jugador -> partido -> posición fija")
    
    # Agrupar participaciones por jugador y partido para extraer su posición típica
    jugador_partido_pos = []
    
    # Usar participaciones para crear mapeo único jugador-partido-equipo-posición
    for (id_jugador, id_punto), group in df_participaciones.groupby(['id_jugador', 'id_punto']):
        # Obtener id_partido del punto
        id_partido = df_puntos[df_puntos['id_punto'] == id_punto]['id_partido'].values[0]
        
        # Hay solo 1 registro por jugador-punto
        record = group.iloc[0]
        
        jugador_partido_pos.append({
            'id_jugador': id_jugador,
            'id_partido': id_partido,
            'equipo': record['equipo'],
            'posicion': record['posicion'].strip(),
            'posicion_equipo': record['posicion_equipo'].strip(),
            'nombre_jugador': df_jugadores[df_jugadores['id_jugador'] == id_jugador]['nombre_completo'].values[0]
        })
    
    df_jugador_partido_pos = pd.DataFrame(jugador_partido_pos)
    
    # Eliminar duplicados (un jugador puede tener la misma posición en todos los puntos del partido)
    df_jugador_partido_pos = df_jugador_partido_pos.drop_duplicates(
        subset=['id_jugador', 'id_partido', 'equipo', 'posicion']
    ).reset_index(drop=True)
    
    print(f"   ✓ {len(df_jugador_partido_pos)} combinaciones únicas jugador-partido-posición")
    
    # Guardar
    jugador_partido_pos_path = os.path.join(output_dir, 'jugador_partido_posicion.csv')
    df_jugador_partido_pos.to_csv(jugador_partido_pos_path, index=False, encoding='utf-8-sig')
    print(f"   ✓ Guardado en: {jugador_partido_pos_path}")
    
    # ==========================================
    # RESUMEN FINAL
    # ==========================================
    print("\n" + "="*70)
    print("✅ NORMALIZACIÓN COMPLETADA")
    print("="*70)
    print(f"\n📁 Archivos generados en: {output_dir}/")
    print(f"   1. jugadores.csv                  - {len(df_jugadores)} jugadores")
    print(f"   2. partidos.csv                   - {len(df_partidos)} partidos")
    print(f"   3. puntos.csv                     - {len(df_puntos)} puntos")
    print(f"   4. participaciones_jugadores.csv  - {len(df_participaciones)} participaciones (FK única)")
    print(f"   5. jugador_partido_posicion.csv   - {len(df_jugador_partido_pos)} combinaciones únicas")
    
    print(f"\n🔑 Llaves Foráneas (Foreign Keys):")
    print(f"   • puntos.id_partido → partidos.id_partido")
    print(f"   • participaciones_jugadores.id_jugador → jugadores.id_jugador")
    print(f"   • participaciones_jugadores.id_punto → puntos.id_punto")
    print(f"   • jugador_partido_posicion.id_jugador → jugadores.id_jugador")
    print(f"   • jugador_partido_posicion.id_partido → partidos.id_partido")
    
    print(f"\n💡 Ejemplo de FK única:")
    print(f"   Para Partido 27, Punto 1, Nicole Lizarazo en IZQUIERDA:")
    
    # Ejemplo
    ejemplo_jugador = df_jugadores[df_jugadores['nombre_completo'] == 'Nicole Lizarazo']['id_jugador'].values
    if len(ejemplo_jugador) > 0:
        id_jugador_ejemplo = ejemplo_jugador[0]
        ejemplo_punto_key = (27, 1)
        if ejemplo_punto_key in punto_key_to_id:
            id_punto_ejemplo = punto_key_to_id[ejemplo_punto_key]
            ejemplo_part = df_participaciones[
                (df_participaciones['id_jugador'] == id_jugador_ejemplo) &
                (df_participaciones['id_punto'] == id_punto_ejemplo)
            ]
            if len(ejemplo_part) > 0:
                ejemplo_id = ejemplo_part.iloc[0]['id_participacion']
                print(f"   → id_participacion = {ejemplo_id}")
                print(f"   → id_jugador = {id_jugador_ejemplo} (Nicole Lizarazo)")
                print(f"   → id_punto = {id_punto_ejemplo} (Partido 27, Punto 1)")
                print(f"   → posicion = {ejemplo_part.iloc[0]['posicion']}")
    
    print("\n" + "="*70)
    
    return {
        'jugadores': df_jugadores,
        'partidos': df_partidos,
        'puntos': df_puntos,
        'participaciones': df_participaciones,
        'jugador_partido_posicion': df_jugador_partido_pos
    }


if __name__ == "__main__":
    # Rutas de archivos
    cancha1_path = "CANCHA1.csv"
    cancha2_path = "CANCHA2.csv"
    output_dir = "data/normalized"
    
    # Ejecutar normalización
    result = normalize_padel_database(cancha1_path, cancha2_path, output_dir)
    
    print("\n🎉 ¡Proceso completado exitosamente!")
    print(f"\n📝 Puedes usar estos archivos para implementar el sistema de mapeo")
    print(f"   con FKs únicas que identifican cada jugador en su contexto específico.")
