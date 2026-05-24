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

    # Crear tabla de Partidas con la columna de capítulo
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS Partidas (
        codigo_partida TEXT PRIMARY KEY,
        unidad TEXT,
        descripcion TEXT,
        capitulo TEXT
    )
    ''')

    conn.commit()
    return conn, cursor

def cargar_datos_json(cursor, json_path, capitulo, mapeo_columnas):
    """Lee el JSON de partidas, inyecta el capítulo y ejecuta un bulk insert/replace en la tabla."""
    if not os.path.exists(json_path):
        print(f"❌ Error: No se encontró el archivo {json_path}")
        return False

    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    if not data:
        print(f"⚠️ Advertencia: {json_path} está vacío.")
        return False

    # Las columnas a insertar incluyen las originales más 'capitulo'
    columnas_completas = mapeo_columnas + ["capitulo"]
    placeholders = ", ".join(["?"] * len(columnas_completas))
    columnas_str = ", ".join(columnas_completas)
    
    query = f"INSERT OR REPLACE INTO Partidas ({columnas_str}) VALUES ({placeholders})"
    
    # Extraer los datos asegurando que el orden coincida y añadiendo el capítulo al final
    registros = []
    for item in data:
        registro = [item.get(col) for col in mapeo_columnas]
        registro.append(capitulo)
        registros.append(registro)
    
    cursor.executemany(query, registros)
    return len(registros)

def main():
    # Obtener el directorio donde se encuentra el script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    db_path = os.path.join(script_dir, "partidas.db")
    
    # Buscar todos los archivos .json en el directorio del script
    patron_json = os.path.join(script_dir, "*.json")
    archivos_json = glob.glob(patron_json)
    
    if not archivos_json:
        print(f"⚠️ No se encontraron archivos .json en: {script_dir}")
        return

    print(f"⚙️ Iniciando migración de todas las partidas a una única base de datos ({os.path.basename(db_path)})...")
    
    try:
        # 1. Inicializar la base de datos única
        conn, cursor = inicializar_base_de_datos(db_path)
        
        columnas = ["codigo_partida", "unidad", "descripcion"]
        total_total = 0
        
        # 2. Procesar cada archivo JSON e insertar sus partidas con el nombre del capítulo
        for json_path in sorted(archivos_json):
            capitulo = os.path.splitext(os.path.basename(json_path))[0]
            print(f"📦 Procesando: {os.path.basename(json_path)} -> Capítulo: '{capitulo}'")
            
            total_insertados = cargar_datos_json(cursor, json_path, capitulo, columnas)
            
            if total_insertados:
                total_total += total_insertados
                print(f"   ✅ Se insertaron {total_insertados} partidas.")
        
        conn.commit()
        conn.close()
        print(f"\n🚀 ¡Migración completada con éxito! Se guardaron {total_total} partidas en '{os.path.basename(db_path)}'.")
        
    except Exception as e:
        print(f"❌ Error durante la migración: {e}")

if __name__ == "__main__":
    main()
