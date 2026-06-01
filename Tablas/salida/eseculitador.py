
import sqlite3
import json
import os

def inicializar_base_de_datos(db_name="covenin.db"):
    """Crea la base de datos y define el esquema relacional estricto."""
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()

    # Es vital activar el chequeo de llaves foráneas en SQLite
    cursor.execute("PRAGMA foreign_keys = ON;")

    # 1. Tabla de Columnas (Diccionario de Categorías)
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS Covenin_Columnas (
        Id_Columna TEXT PRIMARY KEY,
        Nombre TEXT NOT NULL
    )
    ''')

    # 2. Tabla de Valores (El Diccionario Paramétrico)
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS Covenin_Valores (
        Id_Valor TEXT PRIMARY KEY,
        Descripcion_UI TEXT NOT NULL,
        Id_Columna TEXT NOT NULL,
        Num_Min REAL,
        Num_Max REAL,
        Unidad TEXT,
        FOREIGN KEY (Id_Columna) REFERENCES Covenin_Columnas(Id_Columna)
    )
    ''')

    # 3. Tabla de Conexiones (El Grafo / Aristas)
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS Covenin_Conexiones (
        Id_Conexion TEXT PRIMARY KEY,
        Parent_Id TEXT,
        Codigo_Aportado TEXT,
        Id_Columna TEXT NOT NULL,
        Id_Valor_Asociado TEXT,
        FOREIGN KEY (Parent_Id) REFERENCES Covenin_Conexiones(Id_Conexion),
        FOREIGN KEY (Id_Columna) REFERENCES Covenin_Columnas(Id_Columna),
        FOREIGN KEY (Id_Valor_Asociado) REFERENCES Covenin_Valores(Id_Valor)
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

def cargar_datos_json(cursor, ruta_json, nombre_tabla, mapeo_columnas):
    """Lee el JSON y ejecuta un bulk insert en la tabla correspondiente."""
    if not os.path.exists(ruta_json):
        print(f"❌ Error: No se encontró el archivo {ruta_json}")
        return

    with open(ruta_json, 'r', encoding='utf-8') as f:
        data = json.load(f)

    if not data:
        print(f"⚠️ Advertencia: {ruta_json} está vacío.")
        return

    # Preparar la consulta SQL dinámicamente (Evita inyección SQL y maneja nulos)
    placeholders = ", ".join(["?"] * len(mapeo_columnas))
    columnas_str = ", ".join(mapeo_columnas)
    
    # Usamos REPLACE para que si corres el script dos veces, actualice los datos en vez de fallar
    query = f"INSERT OR REPLACE INTO {nombre_tabla} ({columnas_str}) VALUES ({placeholders})"
    
    # Extraer los datos asegurando que el orden coincida con el mapeo de columnas
    registros = [[item.get(col) for col in mapeo_columnas] for item in data]
    
    cursor.executemany(query, registros)
    print(f"✅ Se insertaron {len(registros)} registros en la tabla '{nombre_tabla}'.")

def main():
    nombre_db = "covenin.db"
    print(f"⚙️ Iniciando migración a SQLite ({nombre_db})...")
    
    # 1. Crear esquema
    conn, cursor = inicializar_base_de_datos(nombre_db)

    # 2. Cargar datos en orden jerárquico (primero las dependencias, luego el grafo)
    cargar_datos_json(
        cursor, 
        "tabla_columnas.json", 
        "Covenin_Columnas", 
        ["Id_Columna", "Nombre"]
    )
    
    cargar_datos_json(
        cursor, 
        "tabla_valores.json", 
        "Covenin_Valores", 
        ["Id_Valor", "Descripcion_UI", "Id_Columna", "Num_Min", "Num_Max", "Unidad"]
    )
    
    cargar_datos_json(
        cursor, 
        "tabla_conexiones.json", 
        "Covenin_Conexiones", 
        ["Id_Conexion", "Parent_Id", "Codigo_Aportado", "Id_Columna", "Id_Valor_Asociado"]
    )

    # 3. Guardar y cerrar
    conn.commit()
    conn.close()
    print("\n🚀 ¡Migración a SQLite completada con éxito! La base de datos está lista para Revit.")

if __name__ == "__main__":
    main()