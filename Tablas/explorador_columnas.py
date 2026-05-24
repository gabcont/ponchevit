import json
import os

def extraer_tipos_de_columna(ruta_json_valores):
    """
    Lee el JSON de valores y extrae un set con todos los 'Tipo_Parametro' únicos.
    """
    if not os.path.exists(ruta_json_valores):
        print(f"Error: No se encontró el archivo en {ruta_json_valores}")
        return

    with open(ruta_json_valores, 'r', encoding='utf-8') as f:
        lista_valores = json.load(f)

    # Utilizamos un set comprehension para extraer los tipos únicos rápidamente
    # Ignoramos los valores nulos por seguridad
    tipos_columnas = {
        valor.get("Tipo_Parametro") 
        for valor in lista_valores 
        if valor.get("Tipo_Parametro") is not None
    }

    # Ordenamos alfabéticamente para facilitar la lectura
    tipos_ordenados = sorted(list(tipos_columnas))

    print("=== TIPOS DE COLUMNA DETECTADOS ===")
    print(f"Total únicos: {len(tipos_ordenados)}\n")
    
    for tipo in tipos_ordenados:
        print(f" - {tipo}")
        
    print("\n===================================")

if __name__ == "__main__":
    # Ajusta esta ruta a donde tengas guardado tu JSON de valores
    RUTA_VALORES = "./salida/tabla_valores.json" 
    
    extraer_tipos_de_columna(RUTA_VALORES)