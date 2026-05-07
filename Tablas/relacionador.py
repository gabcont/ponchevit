import os
import json
import time
from typing import Optional

from google import genai
from google.genai import types
from pydantic import BaseModel

api_key = os.environ.get("GEMINI_API_KEY", "")
if not api_key:
    print("Advertencia: No se encontró la variable de entorno GEMINI_API_KEY.")
client = genai.Client(api_key=api_key)

class CabeceraSchema(BaseModel):
    es_vacia: bool
    nombre_columna: Optional[str] = None

class ValorContenido(BaseModel):
    codigo_aporte: Optional[str] = None
    descripcion: str
    med_min: Optional[float] = None
    med_max: Optional[float] = None
    un_medida: Optional[str] = None

class ContenidoSchema(BaseModel):
    es_puente_vacio: bool
    valores: list[ValorContenido] = []

# ==========================================
# 1. DEFINICIÓN DE PROMPTS Y ESQUEMAS
# ==========================================

PROMPT_CABECERA = """
Eres un extractor analítico de datos tabulares. Analiza esta imagen que corresponde al título de una columna.
Ten en cuenta que el recorte puede ser imperfecto y mostrar líneas negras en los bordes; ignora esas líneas de la cuadrícula.

Reglas:
1. Extrae el texto y devuélvelo en mayúsculas como el nombre de la columna.
2. Si la imagen está totalmente en blanco, solo tiene ruido visual/líneas, o no contiene letras legibles, marca 'es_vacia' como true y deja el nombre_columna como null.

Devuelve estrictamente un JSON con esta estructura:
{
  "es_vacia": boolean,
  "nombre_columna": string | null
}
"""

PROMPT_CONTENIDO = """
Eres un extractor de datos técnicos para ingeniería. Analiza esta celda de una tabla normativa.
Ignora cualquier línea negra gruesa en los bordes producto de un recorte imperfecto de la imagen.

Sigue estas reglas estrictamente:
1. Símbolos de vacío: Si la celda está en blanco o solo contiene símbolos como '∅', '-', '/', o marcas de iteración, marca 'es_puente_vacio' como true y devuelve la lista 'valores' vacía.
2. Formato estándar: Extrae la información en formato 'Código: Descripción' (ej. '01: Bloques huecos'). La descripción del código puede estar compuesta por varias líneas de texto.
3. Multi-valores: Una celda puede contener varios valores en lista. Extrae CADA UNO de ellos como un objeto dentro del arreglo 'valores'.
4. Ceros de relleno: Si el código numérico es solo ceros (ej. '0', '00', '0000'), extrae esos ceros en 'codigo_aporte' y pon la descripción como 'Sin Especificar'.
5. Extracción Paramétrica (Numérica):
   - Aísla las medidas numéricas. Ignora variables y signos de igualdad (Ej: si dice 'e = 15 cm' o 'e - 15 cm', asume que el valor es 15 y la unidad 'cm').
   - Si el valor es exacto (Ej: '15 cm'), los campos 'med_min' y 'med_max' deben tener el MISMO valor (15.0).
   - Si el valor es un rango (Ej: '10 - 20 cm'), asigna 'med_min' (10.0) y 'med_max' (20.0).
   - Si no hay medidas, deja esos campos en null.

Devuelve estrictamente un JSON con esta estructura:
{
  "es_puente_vacio": boolean,
  "valores": [
    {
      "codigo_aporte": string | null,
      "descripcion": string,
      "med_min": float | null,
      "med_max": float | null,
      "un_medida": string | null
    }
  ]
}
"""

# ==========================================
# 2. FUNCIÓN ABSTRAÍDA DE IA (Para implementar)
# ==========================================

def extraer_data_con_gemini(ruta_imagen: str, tipo_prompt: str) -> dict:
    """
    Conecta con la API de Gemini para extraer los datos de la imagen recortada.
    
    Args:
        ruta_imagen (str): Ruta local absoluta o relativa a la imagen PNG.
        tipo_prompt (str): Puede ser 'CABECERA' o 'CONTENIDO'.
        
    Returns:
        dict: El JSON parseado que devuelve Gemini según el esquema solicitado.
    """
    prompt = PROMPT_CABECERA if tipo_prompt == 'CABECERA' else PROMPT_CONTENIDO
    schema = CabeceraSchema if tipo_prompt == 'CABECERA' else ContenidoSchema
    
    with open(ruta_imagen, 'rb') as f:
        file_bytes = f.read()

    # Llamada a la API usando el patrón de extractor_manual.py
    for intento in range(3):
        try:
            response = client.models.generate_content(
                model="gemini-3.1-flash-lite-preview",
                contents=[
                    types.Part.from_bytes(
                        data=file_bytes,
                        mime_type="image/png",
                    ),
                    prompt
                ],
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=schema,
                    temperature=0.0
                )
            )
            time.sleep(1) # Control de Rate Limit suave
            return json.loads(response.text)
        except Exception as e:
            print(f"[API ERROR] Intento {intento+1} fallido para {ruta_imagen}: {e}")
            time.sleep(3)
            
    raise Exception(f"Fallo crítico al procesar {ruta_imagen} con Gemini tras 3 intentos.")

# ==========================================
# 3. LÓGICA PRINCIPAL DEL SCRIPT
# ==========================================

def procesar_carpeta_tabla(ruta_carpeta: str):
    """
    Lee el reporte_celdas.json, itera sobre las imágenes de las celdas,
    llama a la IA y guarda un JSON combinado con toda la estructura y datos.
    """
    nombre_carpeta = os.path.basename(os.path.normpath(ruta_carpeta))
    ruta_json_entrada = os.path.join(ruta_carpeta, "reporte_celdas.json")
    ruta_json_salida = f"{nombre_carpeta}_extraido.json"

    if not os.path.exists(ruta_json_entrada):
        print(f"Error: No se encontró el archivo {ruta_json_entrada}")
        return

    # Cargar la estructura base
    with open(ruta_json_entrada, 'r', encoding='utf-8') as f:
        estructura_columnas = json.load(f)

    total_imagenes = sum(columna['num_celdas'] for columna in estructura_columnas)
    procesadas = 0

    print(f"Iniciando procesamiento de '{nombre_carpeta}'. Total imágenes estimadas: {total_imagenes}")

    # Iterar sobre las columnas y sus celdas
    for columna in estructura_columnas:
        id_columna = columna['columna']
        
        # Saltamos la columna 00 por tus indicaciones (solo tiene "E4" y el título)
        if id_columna == 0:
            print("Saltando columna 0 (Raíz del capítulo)...")
            continue

        for celda in columna['celdas']:
            ruta_imagen_relativa = celda['imagen']
            ruta_imagen_absoluta = os.path.join(ruta_carpeta, ruta_imagen_relativa)
            
            # Determinar si es cabecera (celda 0) o contenido
            es_cabecera = (celda['celda'] == 0)
            tipo_prompt = 'CABECERA' if es_cabecera else 'CONTENIDO'

            if os.path.exists(ruta_imagen_absoluta):
                try:
                    # Llamada a la API
                    datos_extraidos = extraer_data_con_gemini(ruta_imagen_absoluta, tipo_prompt)
                    
                    # Inyectar los datos leídos directamente en el nodo del JSON
                    celda['datos_ia'] = datos_extraidos
                    
                except Exception as e:
                    print(f"Error procesando imagen {ruta_imagen_relativa}: {e}")
                    celda['datos_ia'] = {"error": str(e)}
            else:
                print(f"Advertencia: Imagen no encontrada -> {ruta_imagen_absoluta}")
                celda['datos_ia'] = {"error": "Archivo no encontrado"}

            procesadas += 1
            print(f"Progreso: {procesadas}/{total_imagenes} procesadas...", end='\r')

    print("\nProcesamiento con IA finalizado. Guardando archivo unificado...")

    # Guardar el JSON resultante, enriquecido con la data extraída
    with open(ruta_json_salida, 'w', encoding='utf-8') as f_out:
        json.dump(estructura_columnas, f_out, indent=2, ensure_ascii=False)
        
    print(f"Éxito: Archivo generado -> {ruta_json_salida}")

# ==========================================
# PUNTO DE ENTRADA
# ==========================================
if __name__ == "__main__":
    # Cambia "01" por la ruta real a tu carpeta de trabajo
    carpeta_objetivo = "./Extraccion/14" 
    
    procesar_carpeta_tabla(carpeta_objetivo)