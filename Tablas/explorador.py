import json
import os

# Rutas a tus archivos generados
RUTA_VALORES = "./salida/tabla_valores.json"
RUTA_CONEXIONES = "./salida/tabla_conexiones.json"

def main():
    print("Iniciando Explorador de Nodos COVENIN...")
    
    if not os.path.exists(RUTA_VALORES) or not os.path.exists(RUTA_CONEXIONES):
        print("❌ Error: No se encontraron los archivos JSON en la carpeta './salida/'.")
        return

    # 1. Cargar datos
    with open(RUTA_VALORES, 'r', encoding='utf-8') as f:
        lista_valores = json.load(f)
    with open(RUTA_CONEXIONES, 'r', encoding='utf-8') as f:
        lista_conexiones = json.load(f)

    # Diccionario rápido para buscar valores
    valores = {v["Id_Valor"]: v for v in lista_valores}
    
    # 2. Indexar el grafo
    hijos = {}  # Parent_Id -> Lista de conexiones hijas
    nodos = {}  # Id_Conexion -> Datos de la conexión
    
    for conn in lista_conexiones:
        id_conn = conn["Id_Conexion"]
        nodos[id_conn] = conn
        padre = conn["Parent_Id"]
        
        if padre not in hijos:
            hijos[padre] = []
        hijos[padre].append(conn)

    # 3. Mapear códigos acumulados a Nodos
    # Esto nos permite buscar "E411" y saber exactamente en qué nodo del grafo estamos
    codigo_a_nodos = {} 

    def propagar_codigo(nodo_id, codigo_acumulado):
        conn = nodos[nodo_id]
        aporte = conn.get("Codigo_Aportado") or ""
        nuevo_codigo = codigo_acumulado + str(aporte)
        codigo_limpio = nuevo_codigo.replace(".", "") # Limpiar puntos por si acaso

        if codigo_limpio not in codigo_a_nodos:
            codigo_a_nodos[codigo_limpio] = []
        codigo_a_nodos[codigo_limpio].append(nodo_id)

        # Llamada recursiva a los hijos
        for hijo in hijos.get(nodo_id, []):
            propagar_codigo(hijo["Id_Conexion"], nuevo_codigo)

    # Buscar todas las raíces (nodos sin padre) e iniciar la propagación
    raices = [c["Id_Conexion"] for c in lista_conexiones if c["Parent_Id"] is None]
    for r in raices:
        propagar_codigo(r, "")

    print(f"✅ Grafo indexado con éxito. Se encontraron {len(codigo_a_nodos)} rutas únicas.\n")
    print("="*60)
    print(" ESCRIBE UN CÓDIGO PARA VER SUS OPCIONES (Ej: 'E4', 'E411')")
    print(" Escribe 'salir' para terminar.")
    print("="*60)

    # 4. Bucle interactivo
    while True:
        busqueda = input("\nCódigo parcial > ").strip().upper()
        if busqueda == 'SALIR':
            break
        
        if busqueda not in codigo_a_nodos:
            print(f"  ❌ No existe ningún camino en el grafo para el código '{busqueda}'")
            continue

        nodos_encontrados = codigo_a_nodos[busqueda]
        
        # Recopilar todos los hijos posibles de estos nodos
        opciones_siguientes = []
        for n_id in nodos_encontrados:
            for hijo in hijos.get(n_id, []):
                aporte = hijo.get("Codigo_Aportado") or "[Vacío]"
                id_valor = hijo.get("Id_Valor_Asociado")
                
                # Buscar la descripción en la tabla de valores
                desc = "Sin descripción"
                if id_valor and id_valor in valores:
                    desc = valores[id_valor]["Descripcion_UI"]
                
                col_tipo = hijo.get("Columna_Tipo", "Desconocido")
                
                opciones_siguientes.append({
                    "aporte": str(aporte),
                    "desc": desc,
                    "tipo": col_tipo
                })

        # Mostrar resultados
        if not opciones_siguientes:
            print(f"  🏁 El código '{busqueda}' es una HOJA. No hay más ramas a partir de aquí.")
        else:
            print(f"  🔍 Opciones siguientes para '{busqueda}':")
            # Usar un set para evitar imprimir duplicados visuales si distintas ramas convergen
            vistas = set()
            for op in opciones_siguientes:
                firma = f"{op['aporte']}-{op['desc']}"
                if firma not in vistas:
                    vistas.add(firma)
                    # Formateo de tabla simple para la consola
                    print(f"    ├── + {op['aporte'].ljust(4)} | {op['tipo'].ljust(16)} | {op['desc']}")

if __name__ == "__main__":
    main()