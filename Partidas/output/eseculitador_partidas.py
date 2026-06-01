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

    # Crear tabla de Capitulos
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS Capitulos (
        id TEXT PRIMARY KEY,
        codigo TEXT,
        titulo TEXT
    )
    ''')

    # Crear tabla de Subcapitulos
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS Subcapitulos (
        id TEXT PRIMARY KEY,
        capitulo_id TEXT,
        codigo TEXT,
        titulo TEXT,
        FOREIGN KEY (capitulo_id) REFERENCES Capitulos(id)
    )
    ''')

    # Crear tabla de Secciones
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS Secciones (
        id TEXT PRIMARY KEY,
        capitulo_id TEXT,
        subcapitulo_id TEXT,
        codigo TEXT,
        titulo TEXT,
        FOREIGN KEY (capitulo_id) REFERENCES Capitulos(id),
        FOREIGN KEY (subcapitulo_id) REFERENCES Subcapitulos(id)
    )
    ''')

    # Crear tabla de Partidas con la columna de capítulo
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS Partidas (
        codigo_partida TEXT PRIMARY KEY,
        unidad TEXT,
        descripcion TEXT,
        capitulo TEXT
    )
    ''')

    cursor.execute('''
    CREATE TABLE IF NOT EXISTS _meta (
        schema_version INTEGER NOT NULL
    )
    ''')
    cursor.execute("INSERT OR IGNORE INTO _meta (schema_version) SELECT 1 WHERE NOT EXISTS (SELECT 1 FROM _meta)")

    conn.commit()
    return conn, cursor

def cargar_datos_simples(cursor, json_path, nombre_tabla, columnas):
    """Lee el JSON y ejecuta un bulk insert/replace en la tabla especificada."""
    if not os.path.exists(json_path):
        print(f"❌ Error: No se encontró el archivo {json_path}")
        return False

    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    if not data:
        return 0

    placeholders = ", ".join(["?"] * len(columnas))
    columnas_str = ", ".join(columnas)
    
    query = f"INSERT OR REPLACE INTO {nombre_tabla} ({columnas_str}) VALUES ({placeholders})"
    
    registros = [[item.get(col) for col in columnas] for item in data]
    cursor.executemany(query, registros)
    return len(registros)

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
        
        # 1.5 Cargar tablas relacionadas
        path_cap = os.path.join(script_dir, "capitulos.json")
        path_sub = os.path.join(script_dir, "subcapitulos.json")
        path_sec = os.path.join(script_dir, "secciones.json")
        
        c_insertados = cargar_datos_simples(cursor, path_cap, "Capitulos", ["id", "codigo", "titulo"])
        print(f"📦 Capítulos insertados: {c_insertados}")
        
        s_insertados = cargar_datos_simples(cursor, path_sub, "Subcapitulos", ["id", "capitulo_id", "codigo", "titulo"])
        print(f"📦 Subcapítulos insertados: {s_insertados}")
        
        se_insertados = cargar_datos_simples(cursor, path_sec, "Secciones", ["id", "capitulo_id", "subcapitulo_id", "codigo", "titulo"])
        print(f"📦 Secciones insertadas: {se_insertados}")

        columnas = ["codigo_partida", "unidad", "descripcion"]
        total_total = 0
        
        archivos_ignorar = ["capitulos.json", "subcapitulos.json", "secciones.json", "indice.json"]

        # 2. Procesar cada archivo JSON e insertar sus partidas con el nombre del capítulo
        for json_path in sorted(archivos_json):
            nombre_archivo = os.path.basename(json_path)
            if nombre_archivo in archivos_ignorar:
                continue
                
            capitulo = os.path.splitext(nombre_archivo)[0]
            print(f"📦 Procesando: {nombre_archivo} -> Capítulo: '{capitulo}'")
            
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
