import os
import json
import glob
from collections import defaultdict

# ==========================================
# 1. CONFIGURACIÓN DE RUTAS
# ==========================================
DIR_ENTRADA = "./entrada"
DIR_SALIDA = "./salida"
DIR_LISTADO = "./listado"

# Generadores de IDs globales para unificar múltiples archivos
contador_valores = 1
contador_conexiones = 1

def generar_id_valor():
    global contador_valores
    id_val = f"VAL_{contador_valores:05d}"
    contador_valores += 1
    return id_val

def generar_id_conexion():
    global contador_conexiones
    id_conn = f"CON_{contador_conexiones:06d}"
    contador_conexiones += 1
    return id_conn

# ==========================================
# 2. ENSAMBLAJE MASIVO DEL GRAFO
# ==========================================
def procesar_archivos_entrada():
    """
    Lee todos los JSON de la carpeta de entrada y unifica el grafo.
    """
    archivos = glob.glob(os.path.join(DIR_ENTRADA, "*.json"))
    if not archivos:
        print(f"Error: No hay archivos JSON en '{DIR_ENTRADA}'")
        return None, None

    tabla_valores = {}
    tabla_conexiones = []
    
    # Creamos un único nodo raíz para todo el proyecto E4
    id_conexion_raiz_global = generar_id_conexion()
    tabla_conexiones.append({
        "Id_Conexion": id_conexion_raiz_global,
        "Parent_Id": None,
        "Codigo_Aportado": "E4",
        "Columna_Tipo": "CAPITULO",
        "Id_Valor_Asociado": None
    })

    for ruta_archivo in archivos:
        with open(ruta_archivo, 'r', encoding='utf-8') as f:
            datos_tabla = json.load(f)
            
        mapa_rastreo = {}
        # Inicializar la tabla asumiendo que arranca desde el E4 (Columna 0, Fila 1)
        mapa_rastreo[(0, 1)] = [{
            'id_conexion_padre': id_conexion_raiz_global,
            'codigo_acumulado': "E4"
        }]

        for columna in datos_tabla:
            id_columna = columna['columna']
            if id_columna == 0: continue

            celda_cabecera = next((c for c in columna['celdas'] if c['celda'] == 0), None)
            if not celda_cabecera or not celda_cabecera.get('datos_ia'): continue
            nombre_columna = celda_cabecera['datos_ia'].get('nombre_columna', "DESCONOCIDO")

            for celda in columna['celdas']:
                id_celda = celda['celda']
                if id_celda == 0: continue
                
                adyacentes = celda.get('adyacentes_izquierda', [])
                datos_ia = celda.get('datos_ia', {})
                
                conexiones_heredadas = []
                for ady in adyacentes:
                    conexiones_heredadas.extend(mapa_rastreo.get((id_columna - 1, ady), []))

                # Manejo de puente vacío
                if datos_ia.get('es_puente_vacio', False):
                    mapa_rastreo[(id_columna, id_celda)] = conexiones_heredadas
                    continue

                valores = datos_ia.get('valores', [])
                nuevas_conexiones_activas = []

                for val in valores:
                    # Corrección 1: Forzar string vacío si la IA devuelve null/None
                    aporte_crudo = val.get('codigo_aporte')
                    codigo_aporte = str(aporte_crudo).strip() if aporte_crudo is not None else ""
                    
                    # Corrección 2: ELIMINAR el "if codigo_aporte == '00000': continue"
                    # En tu JSON (Celda 4, Panelas de Yeso), el aporte "0000" ES el código 
                    # legítimo para rellenar los vacantes y llegar a 10 dígitos. 
                    # No debemos ignorar los ceros.
                    
                    firma_valor = f"{val.get('descripcion')}|{val.get('med_min')}|{val.get('un_medida')}"
                    if firma_valor not in tabla_valores:
                        tabla_valores[firma_valor] = {
                            "Id_Valor": generar_id_valor(),
                            "Descripcion_UI": val.get('descripcion'),
                            "Tipo_Parametro": nombre_columna,
                            "Num_Min": val.get('med_min'),
                            "Num_Max": val.get('med_max'),
                            "Unidad": val.get('un_medida')
                        }
                    
                    id_valor_asignado = tabla_valores[firma_valor]["Id_Valor"]

                    for padre in conexiones_heredadas:
                        nuevo_codigo_acumulado = padre['codigo_acumulado'] + codigo_aporte
                        
                        # CORTAFUEGOS: Detener caminos mayores a 10 caracteres
                        if len(nuevo_codigo_acumulado.replace(".", "")) > 10:
                            continue 
                        
                        id_nueva_conexion = generar_id_conexion()
                        tabla_conexiones.append({
                            "Id_Conexion": id_nueva_conexion,
                            "Parent_Id": padre['id_conexion_padre'],
                            "Codigo_Aportado": codigo_aporte,
                            "Columna_Tipo": nombre_columna,
                            "Id_Valor_Asociado": id_valor_asignado
                        })
                        
                        nuevas_conexiones_activas.append({
                            'id_conexion_padre': id_nueva_conexion,
                            'codigo_acumulado': nuevo_codigo_acumulado
                        })

                mapa_rastreo[(id_columna, id_celda)] = nuevas_conexiones_activas

    lista_valores_final = list(tabla_valores.values())
    return lista_valores_final, tabla_conexiones

# ==========================================
# 3. ALGORITMO DFS Y VALIDACIÓN
# ==========================================
def generar_todos_los_codigos_posibles(tabla_conexiones):
    """
    Recorre el grafo completo y devuelve un SET con todos los códigos armados.
    """
    hijos = defaultdict(list)
    nodos = {}
    raices = []

    for conn in tabla_conexiones:
        id_conn = conn["Id_Conexion"]
        nodos[id_conn] = conn
        if conn["Parent_Id"] is None:
            raices.append(id_conn)
        else:
            hijos[conn["Parent_Id"]].append(id_conn)

    codigos_generados = set()

    def dfs(nodo_id, codigo_acumulado):
        conn = nodos[nodo_id]
        aporte = conn["Codigo_Aportado"] or ""
        nuevo_codigo = codigo_acumulado + aporte
        
        # Limpiamos puntos por si en las tablas venían códigos como "E4.1"
        nuevo_codigo_limpio = nuevo_codigo.replace(".", "")

        # Si llegamos a 10 dígitos o es un nodo final (hoja)
        es_hoja = len(hijos[nodo_id]) == 0
        if len(nuevo_codigo_limpio) >= 10:
            codigos_generados.add(nuevo_codigo_limpio[:10])
            return # Termina la rama
        
        if es_hoja:
            codigos_generados.add(nuevo_codigo_limpio)
        else:
            for hijo_id in hijos[nodo_id]:
                dfs(hijo_id, nuevo_codigo)

    for raiz in raices:
        dfs(raiz, "")

    return codigos_generados

def ejecutar_prueba_de_fuego(codigos_posibles_set):
    archivos_listado = glob.glob(os.path.join(DIR_LISTADO, "*.json"))
    if not archivos_listado:
        print(f"Advertencia: No se encontró el JSON del listado en '{DIR_LISTADO}'")
        return

    ruta_listado = archivos_listado[0]
    with open(ruta_listado, 'r', encoding='utf-8') as f:
        listado_oficial = json.load(f)

    partidas_fallidas = []
    partidas_exitosas = 0

    for partida in listado_oficial:
        cod_oficial = str(partida.get("codigo_partida", "")).strip().replace(".", "")
        if cod_oficial in codigos_posibles_set:
            partidas_exitosas += 1
        else:
            partidas_fallidas.append(partida)

    # Generar reporte
    ruta_reporte = os.path.join(DIR_SALIDA, "reporte_validacion.json")
    resumen = {
        "Total_Listado": len(listado_oficial),
        "Exitosos_Generados": partidas_exitosas,
        "Fallidos_No_Encontrados": len(partidas_fallidas),
        "Detalle_Fallidos": partidas_fallidas
    }

    with open(ruta_reporte, 'w', encoding='utf-8') as f:
        json.dump(resumen, f, indent=2, ensure_ascii=False)

    print("\n--- RESULTADO DE LA PRUEBA DE FUEGO ---")
    print(f"Total Partidas en el Listado: {len(listado_oficial)}")
    print(f"Partidas Replicadas con Éxito: {partidas_exitosas}")
    print(f"Partidas No Encontradas (Fallos/Errores de Norma): {len(partidas_fallidas)}")
    print(f"Reporte completo guardado en: {ruta_reporte}\n")

# ==========================================
# EJECUCIÓN PRINCIPAL
# ==========================================
if __name__ == "__main__":
    os.makedirs(DIR_SALIDA, exist_ok=True)
    
    print("1. Extrayendo e integrando datos...")
    valores, conexiones = procesar_archivos_entrada()
    
    if valores and conexiones:
        # Exportar BD a JSON
        with open(os.path.join(DIR_SALIDA, 'tabla_valores.json'), 'w', encoding='utf-8') as f:
            json.dump(valores, f, indent=2, ensure_ascii=False)
        with open(os.path.join(DIR_SALIDA, 'tabla_conexiones.json'), 'w', encoding='utf-8') as f:
            json.dump(conexiones, f, indent=2, ensure_ascii=False)
        print("-> Tablas de base de datos exportadas a JSON.")

        print("2. Generando Permutaciones Matemáticas (Grafo DFS)...")
        codigos_generados = generar_todos_los_codigos_posibles(conexiones)
        #print(codigos_generados)
        print(f"-> El Grafo puede generar {len(codigos_generados)} códigos únicos válidos.")

        print("3. Ejecutando Prueba de Fuego...")
        ejecutar_prueba_de_fuego(codigos_generados)