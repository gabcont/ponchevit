import json

def procesar_indice(archivo_entrada):
    # Cargar los datos del archivo JSON
    with open(archivo_entrada, 'r', encoding='utf-8') as f:
        datos = json.load(f)

    # Listas para guardar los elementos clasificados
    capitulos = []
    subcapitulos = []
    secciones = []

    # Contadores para generar los IDs incrementales
    contador_cap = 1
    contador_subcap = 1
    contador_sec = 1

    # Variables para rastrear los IDs actuales y armar las relaciones
    cap_actual_id = None
    subcap_actual_id = None

    for item in datos:
        tipo = item.get("tipo")
        codigo = item.get("codigo")
        titulo = item.get("titulo")

        if tipo == "capitulo":
            # Generar ID (ej: CAP_001)
            cap_id = f"CAP_{contador_cap:03d}"
            cap_actual_id = cap_id  # Actualizar el capítulo actual
            
            capitulos.append({
                "id": cap_id,
                "codigo": codigo,
                "titulo": titulo
            })
            contador_cap += 1

        elif tipo == "subcapitulo":
            # Generar ID (ej: SUBCAP_001)
            subcap_id = f"SUBCAP_{contador_subcap:03d}"
            subcap_actual_id = subcap_id  # Actualizar el subcapítulo actual
            
            subcapitulos.append({
                "id": subcap_id,
                "capitulo_id": cap_actual_id,
                "codigo": codigo,
                "titulo": titulo
            })
            contador_subcap += 1

        elif tipo == "seccion":
            # Generar ID (ej: SEC_001)
            sec_id = f"SEC_{contador_sec:03d}"
            
            secciones.append({
                "id": sec_id,
                "capitulo_id": cap_actual_id,
                "subcapitulo_id": subcap_actual_id,
                "codigo": codigo,
                "titulo": titulo
            })
            contador_sec += 1

    # Guardar los resultados en sus respectivos archivos JSON
    guardar_json('./capitulos.json', capitulos)
    guardar_json('./subcapitulos.json', subcapitulos)
    guardar_json('./secciones.json', secciones)
    
    print(f"Proceso finalizado. Se generaron:")
    print(f"- {len(capitulos)} capítulos en 'capitulos.json'")
    print(f"- {len(subcapitulos)} subcapítulos en 'subcapitulos.json'")
    print(f"- {len(secciones)} secciones en 'secciones.json'")

def guardar_json(nombre_archivo, datos):
    with open(nombre_archivo, 'w', encoding='utf-8') as f:
        json.dump(datos, f, ensure_ascii=False, indent=2)

# Ejecución del script
if __name__ == "__main__":
    # Asegúrate de tener el json unificado guardado como 'indice_completo.json'
    # en la misma carpeta que este script antes de ejecutarlo.
    archivo_origen = './indice.json' 
    
    try:
        procesar_indice(archivo_origen)
    except FileNotFoundError:
        print(f"Error: No se encontró el archivo '{archivo_origen}'.")