import os
import json
import csv
import time
import glob
from google import genai
from google.genai import types
from pydantic import BaseModel, Field

# ==========================================
# 1. CONFIGURACIÓN DEL ENTORNO Y RUTAS
# ==========================================
# Define tus rutas absolutas o relativas aquí
DIR_PDFS = "./Tablas/"         # Carpeta con PDFs de 1 página
DIR_PARTIDAS = "./Listas"   # Carpeta con tus E1.json, E2.json, etc.
DIR_EXTRACCION = "./Extraccion/"  # Salida de la IA
DIR_REPORTES = "./"     # Salida del Linter (CSV)

# Crear directorios si no existen
for d in [DIR_PDFS, DIR_PARTIDAS, DIR_EXTRACCION, DIR_REPORTES]:
    os.makedirs(d, exist_ok=True)

api_key = os.environ.get("GEMINI_API_KEY", "")
if not api_key:
    print("Advertencia: No se encontró la variable de entorno GEMINI_API_KEY.")

# Inicializamos el cliente
client = genai.Client(api_key=api_key)

# ==========================================
# 2. ESQUEMA ESTRICTO PARA LA IA (PYDANTIC)
# ==========================================
class ColumnaAtributo(BaseModel):
    nombre_cabecera: str = Field(
        description="Nombre exacto de la columna en la cabecera (Ej: 'SUB-CAPITULO', 'ACTIVIDAD', 'MECANISMO DE EJECUCION')"
    )
    valor_celda: str = Field(
        description="El texto de la celda. Si estaba visualmente en blanco, coloca el valor heredado de la fila superior (forward-fill)."
    )
    digitos_aportados: str = Field(
        description="Los números que esta celda aporta a la codificación. Ej: si dice '1: Banqueos', aporta '1'. Si la columna dice '(digitos vacantes)' y el valor es '00000', aporta '00000'."
    )

class FilaNorma(BaseModel):
    capitulo_raiz: str = Field(
        description="El capítulo principal al que pertenece la fila (Ej: 'E2 MOVIMIENTO DE TIERRAS')"
    )
    unidad_medida: str = Field(
        description="El valor de la columna 'Un.' (Ej: 'm3', 'm2', 'kgf'). Cadena vacía si en esta fila no aplica."
    )
    ruta_horizontal: list[ColumnaAtributo] = Field(
        description="Recorrido de izquierda a derecha de todas las columnas que componen esta fila."
    )
    codigo_completo_estimado: str = Field(
        description="La concatenación de 'E' + todos los digitos_aportados. DEBE tener exactamente 10 caracteres (Ej: 'E211100000')."
    )

class ResultadoPagina(BaseModel):
    filas_extraidas: list[FilaNorma] = Field(
        description="Lista de todas las filas horizontales válidas extraídas de la tabla."
    )

# Prompt del sistema forzado a comportamiento determinista
SYSTEM_INSTRUCTION = (
    "ERES UN PARSER ESTRUCTURAL ESTRICTO DE LA NORMA COVENIN 2000-92 DE VENEZUELA. "
        "Tu única tarea es leer una tabla escaneada y convertirla en registros relacionales horizontales. "
        "REGLAS CRÍTICAS QUE NO PUEDES ROMPER:\n\n"
        "1. ANATOMÍA DEL CÓDIGO: Todo código de partida COVENIN tiene EXACTAMENTE 10 caracteres. "
        "Empieza con la letra de la especialidad (Ej: 'E') seguida de 9 dígitos numéricos.\n"
        "2. LECTURA HORIZONTAL: Cada fila de la tabla representa una partida o sub-partida. "
        "Debes barrer la tabla de izquierda a derecha, capturando TODAS las columnas.\n"
        "3. HERENCIA (FORWARD-FILL): Las celdas en blanco debajo de un texto significan que heredan ese texto. "
        "NUNCA dejes una celda en blanco si hay un valor superior del cual heredar.\n"
        "4. LA UNIDAD DE MEDIDA: La columna 'Un.' (Unidad) es crítica. Nunca la omitas (Ej: m3, m2, kgf).\n"
        "5. DÍGITOS VACANTES (RELLENO): Si una celda o columna contiene '00000', '00', etc., "
        "NO ESTÁ VACÍA. Son los ceros necesarios para completar los 10 caracteres del código. "
        "Debes extraerlos textualmente en 'digitos_aportados'.")

# ==========================================
# 3. MÓDULO: EXTRACCIÓN (LLM)
# ==========================================
def extraer_datos_pdfs():
    print("--- INICIANDO FASE 1: EXTRACCIÓN CON GEMINI ---")
    pdfs = sorted(glob.glob(os.path.join(DIR_PDFS, "*.pdf")))
    
    for pdf_path in pdfs:
        filename = os.path.basename(pdf_path)
        json_path = os.path.join(DIR_EXTRACCION, filename.replace(".pdf", ".json"))
        
        if os.path.exists(json_path):
            print(f"[CACHE] Saltando {filename}, ya procesado.")
            continue
            
        print(f"[API] Procesando {filename}...")
        try:
            with open(pdf_path, 'rb') as f:
                file_bytes = f.read()

            prompt = f"{SYSTEM_INSTRUCTION}\n\nExtrae todas las reglas de codificación de esta tabla."

            response = client.models.generate_content(
                model="gemini-3.1-flash-lite-preview",
                contents=[
                    types.Part.from_bytes(
                        data=file_bytes,
                        mime_type="application/pdf",
                    ),
                    prompt
                ],
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=ResultadoPagina,
                    temperature=0.0
                )
            )
            with open(json_path, 'w', encoding='utf-8') as f:
                f.write(response.text)
            
            time.sleep(4) # Control de Rate Limit duro
        except Exception as e:
            print(f"[ERROR CRÍTICO] Falló {filename}: {e}")

# ==========================================
# 4. MÓDULO: RECOLECCIÓN DEL GROUND TRUTH
# ==========================================
def cargar_partidas_reales():
    print("--- INICIANDO FASE 2: CARGA DE GROUND TRUTH ---")
    codigos_reales = set()
    archivos_json = glob.glob(os.path.join(DIR_PARTIDAS, "*.json"))
    
    for arch in archivos_json:
        with open(arch, 'r', encoding='utf-8') as f:
            data = json.load(f)
            for partida in data:
                codigo = str(partida.get("codigo_partida", "")).strip()
                if codigo:
                    codigos_reales.add(codigo)
                    
    print(f"[INFO] Se cargaron {len(codigos_reales)} códigos reales únicos del tabulador.")
    return codigos_reales

# ==========================================
# 5. MÓDULO: ENSAMBLAJE DEL GRAFO Y DFS
# ==========================================
class NodoGrafo:
    def __init__(self, id_nodo, codigo=""):
        self.id_nodo = id_nodo
        self.codigo = codigo
        self.hijos = {} # Diccionario para evitar aristas duplicadas

def ensamblar_y_recorrer_grafo():
    print("--- INICIANDO FASE 3: CONSTRUCCIÓN DEL GRAFO Y DFS ---")
    raiz_virtual = NodoGrafo("ROOT")
    
    # 5.1 Construir Grafo en Memoria
    archivos_extraidos = glob.glob(os.path.join(DIR_EXTRACCION, "*.json"))
    for arch in archivos_extraidos:
        with open(arch, 'r', encoding='utf-8') as f:
            try:
                data = json.load(f)
                reglas = data.get("reglas_extraidas", [])
            except:
                continue
                
            for regla in reglas:
                ruta_nodos = regla["linaje_padres"] + [regla["valor_actual"]]
                codigo_aporte = regla["codigo_aportado"].strip()
                
                nodo_actual = raiz_virtual
                ruta_acumulada = ""
                
                # Ensamblar la rama iterativamente
                for i, nombre_nodo in enumerate(ruta_nodos):
                    # Normalización agresiva para evitar duplicados por typos de la IA
                    nombre_normalizado = nombre_nodo.strip().upper() 
                    ruta_acumulada += f"|{nombre_normalizado}"
                    
                    if ruta_acumulada not in nodo_actual.hijos:
                        # Solo asignamos el código al nodo hoja de esta iteración
                        aporte = codigo_aporte if i == len(ruta_nodos) - 1 else ""
                        nodo_actual.hijos[ruta_acumulada] = NodoGrafo(ruta_acumulada, aporte)
                    
                    nodo_actual = nodo_actual.hijos[ruta_acumulada]
                    
    # 5.2 DFS (Búsqueda en Profundidad)
    codigos_generados = set()
    
    def dfs(nodo, codigo_acumulado):
        nuevo_codigo = codigo_acumulado + nodo.codigo
        if not nodo.hijos: # Es un nodo hoja (Final de la partida)
            # Solo guardamos si parece un código COVENIN válido (ej. empieza con E y tiene longitud)
            if nuevo_codigo.startswith("E") and len(nuevo_codigo) >= 9:
                codigos_generados.add(nuevo_codigo)
            return
            
        for hijo in nodo.hijos.values():
            dfs(hijo, nuevo_codigo)

    dfs(raiz_virtual, "")
    print(f"[INFO] El Grafo generó {len(codigos_generados)} combinaciones de códigos posibles.")
    return codigos_generados

# ==========================================
# 6. MÓDULO: LINTER Y REPORTE
# ==========================================
def generar_reporte(codigos_reales, codigos_generados):
    print("--- INICIANDO FASE 4: VALIDACIÓN CRUZADA ---")
    
    falsos_positivos = codigos_generados - codigos_reales # El grafo lo armó, pero no existe en el tabulador
    falsos_negativos = codigos_reales - codigos_generados # Existe en el tabulador, pero el grafo no pudo armarlo
    matches = codigos_reales.intersection(codigos_generados)
    
    print(f"Matches perfectos: {len(matches)}")
    print(f"Falsos Positivos (Ramas huérfanas o no usadas): {len(falsos_positivos)}")
    print(f"Falsos Negativos (Ramas faltantes o typos en JSON): {len(falsos_negativos)}")
    
    reporte_path = os.path.join(DIR_REPORTES, "reporte_linter_covenin.csv")
    with open(reporte_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["TIPO_ERROR", "CODIGO", "DESCRIPCION"])
        
        for fp in falsos_positivos:
            writer.writerow(["FALSO_POSITIVO", fp, "Generado por reglas, ausente en tabulador."])
            
        for fn in falsos_negativos:
            writer.writerow(["FALSO_NEGATIVO", fn, "Presente en tabulador, imposible de generar por el grafo."])
            
    print(f"[EXITO] Reporte guardado en: {reporte_path}")

# ==========================================
# EJECUCIÓN PRINCIPAL
# ==========================================
if __name__ == "__main__":
    # 1. Extraer data (Si ya está extraída, usará el caché)
    extraer_datos_pdfs()
    
    # 2. Cargar Ground Truth
    reales = cargar_partidas_reales()
    
    # 3. Ensamblar e iterar el Grafo
    generados = ensamblar_y_recorrer_grafo()
    
    # 4. Cruzar y reportar
    generar_reporte(reales, generados)