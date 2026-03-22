import os
import time
import json
from pathlib import Path
from pydantic import BaseModel

from google import genai
from google.genai import types

# ==========================================
# CONFIGURACIÓN
# ==========================================
ROOT_DIR = "/home/gab/Dev/proyectos/PEG/Partidas/Assets/Paginas/" 
OUTPUT_DIR = "./output/" 

api_key = os.environ.get("GEMINI_API_KEY", "")
if not api_key:
    print("Advertencia: No se encontró la variable de entorno GEMINI_API_KEY.")

# Inicializamos el cliente
client = genai.Client(api_key="")

# Estructura esperada para la salida JSON
class Partida(BaseModel):
    codigo_partida: str
    unidad: str
    descripcion: str

def parse_pdf_with_gemini(file_path: str) -> list[dict]:
    filename = os.path.basename(file_path)
    print(f"  Procesando {filename}...")
    
    # Determinar el mime_type correcto basado en la extensión
    ext = filename.lower().split('.')[-1]
    if ext == 'pdf':
        mime_type = 'application/pdf'
    elif ext in ['jpg', 'jpeg']:
        mime_type = 'image/jpeg'
    elif ext == 'png':
        mime_type = 'image/png'
    else:
        print(f"  [Omitido] Formato no soportado para {filename}")
        return []

    try:
        # Leer el archivo localmente a bytes, tal como en tu ejemplo
        with open(file_path, 'rb') as f:
            file_bytes = f.read()
        
        prompt = (
            "Extrae la tabla de partidas de este documento técnico de la normativa COVENIN. "
            "El documento es una imagen o escaneo que contiene filas con 3 columnas principales: "
            "1) PARTIDA (Código alfanumérico)\n"
            "2) UNIDAD (ej: m2, m, kg, etc.)\n"
            "3) DESCRIPCION (Texto descriptivo)\n\n"
            "INSTRUCCIONES CRÍTICAS:\n"
            "- Extrae cada fila exactamente como aparece.\n"
            "- La 'DESCRIPCION' suele ocupar varias líneas. Únelas en un solo texto continuo, sin saltos de línea.\n"
            "- No inventes códigos ni modifiques el texto."
        )
        
        # Generar contenido enviando los bytes directamente
        response = client.models.generate_content(
            model='gemini-3.1-flash-lite-preview',
            contents=[
                types.Part.from_bytes(
                    data=file_bytes,
                    mime_type=mime_type,
                ),
                prompt
            ],
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=list[Partida],
                temperature=0.0,
            )
        )
        
        try:
            data = json.loads(response.text)
            return data
        except json.JSONDecodeError:
            print(f"  [Error] No se pudo parsear el JSON: {response.text}")
            return []
            
    except Exception as e:
        print(f"  [Error] Falló el procesamiento de {file_path}: {e}")
        return []

def main():
    if not ROOT_DIR or ROOT_DIR == "TU_RUTA_AQUI":
        print("Error: Configura la variable ROOT_DIR.")
        return
        
    root_path = Path(ROOT_DIR)
    if not root_path.exists() or not root_path.is_dir():
        print(f"Error: El directorio {ROOT_DIR} no existe.")
        return

    for capitulo_dir in root_path.iterdir():
        if not capitulo_dir.is_dir():
            continue
            
        nombre_capitulo = capitulo_dir.name
        listado_dir = capitulo_dir / "listado"
        if not listado_dir.exists():
            listado_dir = capitulo_dir / "Listado"
            
        if not listado_dir.exists():
            continue
            
        print(f"\n=========================================")
        print(f"Procesando capítulo: {nombre_capitulo}")
        print(f"=========================================")
        
        todas_las_partidas = []
        archivos_validos = list(listado_dir.glob("*.pdf")) # + list(listado_dir.glob("*.jpg")) + list(listado_dir.glob("*.png"))
        
        for archivo in sorted(archivos_validos):
            partidas_extraidas = parse_pdf_with_gemini(str(archivo))
            if partidas_extraidas:
                todas_las_partidas.extend(partidas_extraidas)
                print(f"  -> Extraídas {len(partidas_extraidas)} partidas de {archivo.name}")
            
            # Pausa recomendada para evitar rate limits
            time.sleep(3)
            
        if todas_las_partidas:
            json_filename = Path(OUTPUT_DIR) / f"{nombre_capitulo}.json"
            with open(json_filename, "w", encoding="utf-8") as f:
                json.dump(todas_las_partidas, f, indent=4, ensure_ascii=False)
            print(f"\n✅ Terminado: Guardadas {len(todas_las_partidas)} partidas en '{json_filename}'")

if __name__ == "__main__":
    main()