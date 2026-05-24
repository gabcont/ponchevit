import sqlite3
import json
import os
import glob

def inicializar_base_de_datos(db_path):
    """Crea la base de datos y define la tabla para las partidas."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Habilitar el chequeo de llaves foráneas en SQLite
    cursor.execute("PRAGMA foreign_keys = ON;")

    # Crear tabla de Partidas
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS Partidas (
        codigo_partida TEXT PRIMARY KEY,
        unidad TEXT,
        descripcion TEXT
    )
    ''')

    conn.commit()
    return conn, cursor

def cargar_datos_json(cursor, json_path, mapeo_columnas):
    """Lee el JSON de partidas y ejecuta un bulk insert/replace en la tabla."""
    if not os.path.exists(json_path):
        print(f"❌ Error: No se encontró el archivo {json_path}")
        return False

    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    if not data:
        print(f"⚠️ Advertencia: {json_path} está vacío.")
        return False

    # Preparar la consulta SQL dinámicamente
    placeholders = ", ".join(["?"] * len(mapeo_columnas))
    columnas_str = ", ".join(mapeo_columnas)
    
    # Usamos REPLACE para evitar fallas por duplicados al re-ejecutar
    query = f"INSERT OR REPLACE INTO Partidas ({columnas_str}) VALUES ({placeholders})"
    
    # Extraer los datos asegurando que el orden coincida con el mapeo de columnas
    registros = [[item.get(col) for col in mapeo_columnas] for item in data]
    
    cursor.executemany(query, registros)
    return len(registros)

def main():
    # Obtener el directorio donde se encuentra el script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Buscar todos los archivos .json en el directorio del script
    patron_json = os.path.join(script_dir, "*.json")
    archivos_json = glob.glob(patron_json)
    
    if not archivos_json:
        print(f"⚠️ No se encontraron archivos .json en: {script_dir}")
        return

    print(f"⚙️ Iniciando migración de partidas a SQLite ({len(archivos_json)} archivos encontrados)...")
    
    columnas = ["codigo_partida", "unidad", "descripcion"]
    
    for json_path in sorted(archivos_json):
        base_name = os.path.splitext(os.path.basename(json_path))[0]
        db_path = os.path.join(script_dir, f"{base_name}.db")
        
        print(f"\n📦 Procesando: {os.path.basename(json_path)} -> {os.path.basename(db_path)}...")
        
        try:
            # 1. Inicializar la base de datos
            conn, cursor = inicializar_base_de_datos(db_path)
            
            # 2. Cargar y guardar datos
            total_insertados = cargar_datos_json(cursor, json_path, columnas)
            
            if total_insertados:
                conn.commit()
                print(f"✅ Se insertaron {total_insertados} partidas con éxito.")
            
            conn.close()
        except Exception as e:
            print(f"❌ Error al procesar {os.path.basename(json_path)}: {e}")

    print("\n🚀 ¡Migración de partidas a SQLite completada con éxito!")

if __name__ == "__main__":
    main()
