import os
import json
from google import genai
from google.genai import types
from pydantic import BaseModel, Field

# ==========================================
# CONFIGURACIÓN
# ==========================================
api_key = os.environ.get("GEMINI_API_KEY", "AIzaSyB847wR8tkxcpekj7OTzr6Zs89zX5ffqqo")
if not api_key:
    print("Advertencia: No se encontró la variable de entorno GEMINI_API_KEY.")

# Inicializamos el cliente
client = genai.Client(api_key=api_key)
ARCHIVO_PDF = "in.pdf"

# ==========================================
# ESQUEMA ESTRICTO PARA TU IDEA
# ==========================================
class ColumnaDatos(BaseModel):
    nombre_columna: str = Field(
        description="El nombre de la cabecera de la columna."
    )
    valores: list[str] = Field(
        description="Textos leídos de arriba hacia abajo. Usa '*' EXCLUSIVAMENTE cuando cruces una línea física horizontal."
    )

class ExtraccionTabla(BaseModel):
    columnas: list[ColumnaDatos] = Field(
        description="Lista de todas las columnas detectadas."
    )

# ==========================================
# MOTOR DE IA (PROMPT EXPERIMENTAL)
# ==========================================
SYSTEM_INSTRUCTION = (
    "Eres un analizador óptico de documentos rígido. Vas a leer una tabla escaneada que tiene líneas de cuadrícula negras (trazos horizontales y verticales)."
    "\n\nTU ÚNICA TAREA:"
    "\n1. Lee la tabla columna por columna, de izquierda a derecha."
    "\n2. Dentro de cada columna, lee el texto estrictamente de ARRIBA hacia ABAJO."
    "\n3. REGLA DE LA LÍNEA NEGRA: Cada vez que tu lectura vertical cruce una LÍNEA FÍSICA HORIZONTAL IMPRESA (un trazo negro que separa celdas), DEBES insertar el carácter '*' como un elemento independiente en la lista."
    "\n4. NO insertes '*' por espacios en blanco o saltos de párrafo. EL ASTERISCO ES SOLO PARA LÍNEAS NEGRAS HORIZONTALES."
    "\n5. Si una celda tiene varias líneas de texto pero no hay una línea negra que las divida, agrúpalas en un solo string."
)

def ejecutar_prueba(pdf_path):
    print(f"--- INICIANDO PRUEBA DE ASTERISCOS: {pdf_path} ---")
    
    try:
        print("[1] Leyendo archivo PDF...")
        with open(pdf_path, 'rb') as f:
            file_bytes = f.read()

        prompt = f"{SYSTEM_INSTRUCTION}\n\nExtrae el texto de las columnas e inserta '*' solo cuando cruces una línea horizontal impresa."
        
        print("[2] Analizando con Gemini (Lectura Vertical + Asteriscos)...")
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
                response_schema=ExtraccionTabla,
                temperature=0.0 # Temperatura cero para intentar forzar consistencia
            )
        )
        
        # Parseamos el resultado
        data = json.loads(response.text)
        
        # Imprimimos el resultado en la consola para evaluación visual
        print("\n[+] RESULTADO DE LA EXTRACCIÓN:\n")
        for col in data.get("columnas", []):
            print(f"COLUMNA: {col['nombre_columna']}")
            for val in col['valores']:
                if val == "*":
                    pass
                else:
                    # Limpiamos saltos de línea internos para ver el bloque completo
                    texto_limpio = val.replace('\n', ' ').strip()
                    print(f"  - {texto_limpio}")
            print("-" * 40)
            
        print("\n[INFO] Prueba finalizada.")
        
    except Exception as e:
        print(f"[ERROR] La ejecución falló: {e}")

if __name__ == "__main__":
    ejecutar_prueba(ARCHIVO_PDF)