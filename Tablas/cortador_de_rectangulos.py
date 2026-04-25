#!/usr/bin/env python3
"""
cortador_de_rectangulos.py
──────────────────────────
Recibe un PDF de una página (tabla escaneada de la norma COVENIN 2000-92),
lo convierte a imagen, rota 90° en sentido horario, detecta la grilla de la
tabla mediante morfología (OpenCV), y recorta cada celda como imagen
independiente organizada por columna.

Uso:
    python cortador_de_rectangulos.py entrada.pdf --salida ./recortes/ --debug
    python cortador_de_rectangulos.py imagen.png  --salida ./recortes/
"""

import argparse
import json
import os
import sys

import cv2
import numpy as np
from pdf2image import convert_from_path


# ════════════════════════════════════════════
# 1. CARGA Y CONVERSIÓN
# ════════════════════════════════════════════

def cargar_imagen(ruta: str, dpi: int = 300) -> np.ndarray:
    """
    Carga una imagen desde archivo.  Si es PDF, convierte la primera
    página a imagen con el DPI indicado y la rota 90° en sentido horario.
    """
    ext = os.path.splitext(ruta)[1].lower()

    if ext == ".pdf":
        paginas = convert_from_path(ruta, dpi=dpi, first_page=1, last_page=1)
        if not paginas:
            raise RuntimeError(f"No se pudo convertir el PDF: {ruta}")
        # PIL → numpy (BGR para OpenCV)
        img = cv2.cvtColor(np.array(paginas[0]), cv2.COLOR_RGB2BGR)
        # Rotación 90° en sentido horario
        img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    else:
        img = cv2.imread(ruta)
        if img is None:
            raise FileNotFoundError(f"No se pudo leer la imagen: {ruta}")

    return img


# ════════════════════════════════════════════
# 2. PREPROCESAMIENTO
# ════════════════════════════════════════════

def preprocesar(img: np.ndarray) -> np.ndarray:
    """
    Convierte a escala de grises y binariza con umbral adaptativo.
    Retorna imagen binaria donde las líneas negras son 255 (blanco)
    y el fondo es 0 (negro).
    """
    gris = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Umbral adaptativo invertido: líneas oscuras → 255
    binaria = cv2.adaptiveThreshold(
        gris, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        blockSize=15,
        C=10
    )
    return binaria


# ════════════════════════════════════════════
# 2.5 CORRECCIÓN DE INCLINACIÓN (DESKEW)
# ════════════════════════════════════════════

def detectar_angulo(img: np.ndarray) -> float:
    """
    Detecta el ángulo de inclinación de la imagen analizando las
    líneas rectas (bordes de tabla) mediante la transformada de Hough.

    Usa resolución angular fina (0.1°) y pondera cada línea por su
    longitud para que las líneas largas (más fiables) dominen el
    cálculo sobre las cortas.

    Returns:
        Ángulo en grados (positivo = sentido antihorario).
        Rango esperado: -5° a +5°.
    """
    gris = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, bw = cv2.threshold(gris, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Resolución angular fina: 0.1° para captar skew pequeños
    lineas = cv2.HoughLinesP(
        bw, 1, np.pi / 1800,
        threshold=200,
        minLineLength=img.shape[1] // 8,
        maxLineGap=20
    )

    if lineas is None or len(lineas) == 0:
        return 0.0

    angulos = []
    pesos = []
    for linea in lineas:
        x1, y1, x2, y2 = linea[0]
        dx = x2 - x1
        dy = y2 - y1
        longitud = np.sqrt(dx * dx + dy * dy)
        angulo = np.degrees(np.arctan2(dy, dx))

        # Solo considerar líneas casi horizontales (±15°)
        if abs(angulo) <= 15:
            angulos.append(angulo)
            pesos.append(longitud)
        # También considerar líneas casi verticales (90° ± 15°)
        elif abs(abs(angulo) - 90) <= 15:
            desv = angulo - 90 if angulo > 0 else angulo + 90
            angulos.append(desv)
            pesos.append(longitud)

    if not angulos:
        return 0.0

    # Mediana ponderada por longitud de línea
    orden = np.argsort(angulos)
    angulos_ord = np.array(angulos)[orden]
    pesos_ord = np.array(pesos)[orden]
    pesos_acum = np.cumsum(pesos_ord)
    mitad = pesos_acum[-1] / 2
    idx_mediana = np.searchsorted(pesos_acum, mitad)

    return float(angulos_ord[idx_mediana])


def corregir_inclinacion(img: np.ndarray, umbral_correccion: float = 0.1
                         ) -> tuple[np.ndarray, float]:
    """
    Detecta y corrige la inclinación de la imagen escaneada.

    Args:
        img:                  imagen BGR
        umbral_correccion:    ángulo mínimo (°) para aplicar corrección.
                              Si el ángulo es menor, se devuelve la imagen
                              sin modificar (para evitar rotaciones inútiles).

    Returns:
        (imagen_corregida, angulo_detectado)
    """
    angulo = detectar_angulo(img)

    if abs(angulo) < umbral_correccion:
        return img, angulo

    alto, ancho = img.shape[:2]
    centro = (ancho // 2, alto // 2)

    # Rotar la imagen.  El ángulo de cv2 es sentido antihorario,
    # y queremos corregir la inclinación, así que rotamos -angulo.
    M = cv2.getRotationMatrix2D(centro, angulo, 1.0)

    # Calcular nuevo tamaño para no recortar esquinas
    cos_a = abs(M[0, 0])
    sin_a = abs(M[0, 1])
    nuevo_ancho = int(alto * sin_a + ancho * cos_a)
    nuevo_alto = int(alto * cos_a + ancho * sin_a)
    M[0, 2] += (nuevo_ancho - ancho) / 2
    M[1, 2] += (nuevo_alto - alto) / 2

    corregida = cv2.warpAffine(
        img, M, (nuevo_ancho, nuevo_alto),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255)   # fondo blanco
    )

    return corregida, angulo


# ════════════════════════════════════════════
# 3. DETECCIÓN MORFOLÓGICA DE LÍNEAS
# ════════════════════════════════════════════

def detectar_lineas(binaria: np.ndarray, escala: int = 30):
    """
    Detecta líneas horizontales y verticales usando operaciones
    morfológicas con kernels elongados.

    Args:
        binaria: imagen binarizada (líneas=255, fondo=0)
        escala:  divisor del ancho/alto para determinar el largo
                 mínimo de línea detectable.  Valores más altos
                 → se detectan líneas más cortas.

    Returns:
        (mascara_horizontal, mascara_vertical) — dos imágenes binarias
    """
    alto, ancho = binaria.shape

    # ── Líneas horizontales ──
    largo_h = max(ancho // escala, 1)
    kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (largo_h, 1))
    horizontal = cv2.morphologyEx(binaria, cv2.MORPH_OPEN, kernel_h, iterations=2)

    # ── Líneas verticales ──
    largo_v = max(alto // escala, 1)
    kernel_v = cv2.getStructuringElement(cv2.MORPH_RECT, (1, largo_v))
    vertical = cv2.morphologyEx(binaria, cv2.MORPH_OPEN, kernel_v, iterations=1)

    return horizontal, vertical


# ════════════════════════════════════════════
# 4. INTERSECCIONES
# ════════════════════════════════════════════

def encontrar_intersecciones(horizontal: np.ndarray,
                              vertical: np.ndarray) -> list[tuple[int, int]]:
    """
    Encuentra los puntos donde se cruzan líneas horizontales y verticales.
    Dilata ligeramente ambas máscaras para asegurar superposición y luego
    hace AND bitwise.

    Returns:
        Lista de puntos (x, y) correspondientes al centroide de cada
        intersección detectada.
    """
    kernel_dilatar = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    h_dilatada = cv2.dilate(horizontal, kernel_dilatar, iterations=2)
    v_dilatada = cv2.dilate(vertical, kernel_dilatar, iterations=2)

    cruces = cv2.bitwise_and(h_dilatada, v_dilatada)

    # Encontrar los contornos de cada mancha de intersección
    contornos, _ = cv2.findContours(cruces, cv2.RETR_EXTERNAL,
                                     cv2.CHAIN_APPROX_SIMPLE)

    puntos = []
    for cnt in contornos:
        M = cv2.moments(cnt)
        if M["m00"] == 0:
            continue
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        puntos.append((cx, cy))

    return puntos


# ════════════════════════════════════════════
# 5. CONSTRUCCIÓN DE LA GRILLA (PER-COLUMNA)
# ════════════════════════════════════════════

def agrupar_coordenadas(valores: list[int], tolerancia: int = 15) -> list[int]:
    """
    Agrupa coordenadas cercanas (dentro de `tolerancia` pixels) y
    retorna el promedio de cada grupo, ordenado.
    """
    if not valores:
        return []

    valores_ord = sorted(valores)
    grupos = [[valores_ord[0]]]

    for v in valores_ord[1:]:
        if v - grupos[-1][-1] <= tolerancia:
            grupos[-1].append(v)
        else:
            grupos.append([v])

    return [int(np.mean(g)) for g in grupos]


def _asignar_punto_a_columna(px: int, columnas_x: list[int],
                              tolerancia: int) -> int | None:
    """
    Dado un punto X, retorna el índice de la columna X más cercana
    dentro de la tolerancia, o None si no coincide con ninguna.
    """
    for i, cx in enumerate(columnas_x):
        if abs(px - cx) <= tolerancia:
            return i
    return None


def construir_grilla_por_columna(
    puntos: list[tuple[int, int]],
    tolerancia: int = 15,
    modo: str = 'interseccion'
) -> tuple[list[int], dict[int, list[int]]]:
    """
    Construye una grilla donde cada columna tiene sus propias filas Y.

    Args:
        puntos:      lista de (x, y) de intersecciones detectadas
        tolerancia:  pixels de tolerancia para agrupar coordenadas
        modo:        'interseccion' — solo Ys que existen en AMBOS bordes
                         (estricto, evita filas fantasma)
                     'union' — Ys que existen en al menos UN borde
                         (permisivo, para tablas con líneas débiles)

    Returns:
        columnas_x:  lista ordenada de coordenadas X de líneas verticales
        celdas_por_columna:  dict {col_idx: [y0, y1, y2, ...]}
    """
    if not puntos:
        raise ValueError(
            "No se detectaron intersecciones. "
            "¿La imagen tiene una tabla con líneas claras?"
        )

    # 1. Extraer coordenadas únicas de columnas (X)
    xs = [p[0] for p in puntos]
    columnas_x = agrupar_coordenadas(xs, tolerancia)

    if len(columnas_x) < 2:
        raise ValueError(
            f"Solo se detectó {len(columnas_x)} línea vertical. "
            "Se necesitan al menos 2 para formar una columna."
        )

    # 2. Para cada punto, asignarlo a su columna X más cercana
    puntos_por_x: dict[int, list[int]] = {i: [] for i in range(len(columnas_x))}

    for (px, py) in puntos:
        idx = _asignar_punto_a_columna(px, columnas_x, tolerancia)
        if idx is not None:
            puntos_por_x[idx].append(py)

    # 3. Construir filas Y por columna
    celdas_por_columna: dict[int, list[int]] = {}
    n_cols = len(columnas_x) - 1

    for col_idx in range(n_cols):
        ys_izq = sorted(agrupar_coordenadas(puntos_por_x[col_idx], tolerancia))
        ys_der = sorted(agrupar_coordenadas(puntos_por_x[col_idx + 1], tolerancia))

        # Fase A: emparejar Ys con tolerancia (promedio de pares)
        ys_resultado = []
        der_usados = set()
        izq_emparejados = set()

        for i, yi in enumerate(ys_izq):
            mejor_j = None
            mejor_dist = tolerancia + 1
            for j, yd in enumerate(ys_der):
                if j in der_usados:
                    continue
                dist = abs(yi - yd)
                if dist <= tolerancia and dist < mejor_dist:
                    mejor_j = j
                    mejor_dist = dist
            if mejor_j is not None:
                der_usados.add(mejor_j)
                izq_emparejados.add(i)
                ys_resultado.append((yi + ys_der[mejor_j]) // 2)

        # Fase B (solo en modo 'union'): agregar Ys de un solo lado
        if modo == 'union':
            for i, yi in enumerate(ys_izq):
                if i not in izq_emparejados:
                    ys_resultado.append(yi)
            for j, yd in enumerate(ys_der):
                if j not in der_usados:
                    ys_resultado.append(yd)

        ys_resultado.sort()

        if len(ys_resultado) >= 2:
            celdas_por_columna[col_idx] = ys_resultado

    return columnas_x, celdas_por_columna


# ════════════════════════════════════════════
# 6. RECORTE DE CELDAS (PER-COLUMNA)
# ════════════════════════════════════════════

def recortar_celdas(img: np.ndarray,
                    columnas_x: list[int],
                    celdas_por_columna: dict[int, list[int]],
                    dir_salida: str,
                    margen: int = 0) -> dict:
    """
    Recorta cada celda usando los bordes Y específicos de cada columna.

    Estructura de salida:
        dir_salida/
        ├── columna_00_cabecera.png
        ├── columna_00/
        │   ├── fila_01.png
        │   ├── fila_02.png
        │   └── ...
        ├── columna_01_cabecera.png
        ├── columna_01/
        │   └── ...
        └── ...

    Args:
        img:                  imagen original (color)
        columnas_x:           coordenadas X de las líneas verticales
        celdas_por_columna:   dict {col_idx: [y0, y1, ...]} por columna
        dir_salida:           carpeta de salida
        margen:               pixels a recortar por dentro para eliminar bordes

    Returns:
        Diccionario con estadísticas
    """
    os.makedirs(dir_salida, exist_ok=True)
    alto_img, ancho_img = img.shape[:2]
    celdas_total = 0

    for col_idx, filas_y in sorted(celdas_por_columna.items()):
        x1 = columnas_x[col_idx] + margen
        x2 = columnas_x[col_idx + 1] - margen

        x1 = max(0, x1)
        x2 = min(ancho_img, x2)
        if x2 <= x1:
            continue

        nombre_col = f"columna_{col_idx:02d}"
        dir_col = os.path.join(dir_salida, nombre_col)
        os.makedirs(dir_col, exist_ok=True)

        n_filas = len(filas_y) - 1
        for fila_idx in range(n_filas):
            y1 = filas_y[fila_idx] + margen
            y2 = filas_y[fila_idx + 1] - margen

            y1 = max(0, y1)
            y2 = min(alto_img, y2)
            if y2 <= y1:
                continue

            celda = img[y1:y2, x1:x2]

            if fila_idx == 0:
                ruta = os.path.join(dir_salida, f"{nombre_col}_cabecera.png")
            else:
                ruta = os.path.join(dir_col, f"fila_{fila_idx:02d}.png")

            cv2.imwrite(ruta, celda)
            celdas_total += 1

    return {
        "columnas": len(celdas_por_columna),
        "celdas_totales": celdas_total,
    }


# ════════════════════════════════════════════
# 6.5 REPORTE DE ADYACENCIA (JSON)
# ════════════════════════════════════════════

def calcular_adyacencia(
    celdas_por_columna: dict[int, list[int]],
    min_solapamiento: int = 10
) -> list[dict]:
    """
    Para cada celda de cada columna, calcula qué celdas de la columna
    anterior son adyacentes (sus rangos Y se solapan en la frontera
    vertical compartida).

    Args:
        celdas_por_columna: dict {col_idx: [y0, y1, ...]}
        min_solapamiento:   mínimo de pixels de solapamiento Y para
                            considerar dos celdas adyacentes.  Evita
                            falsos positivos por 1-2px de skew.

    Returns:
        Lista de diccionarios con la estructura del JSON de salida.
    """
    columnas_ordenadas = sorted(celdas_por_columna.keys())
    reporte = []

    for col_idx in columnas_ordenadas:
        filas_y = celdas_por_columna[col_idx]
        n_celdas = len(filas_y) - 1

        col_data = {
            "columna": col_idx,
            "num_celdas": n_celdas,
            "celdas": []
        }

        for celda_idx in range(n_celdas):
            y_top = filas_y[celda_idx]
            y_bot = filas_y[celda_idx + 1]

            celda_info = {
                "celda": celda_idx,
                "y_inicio": y_top,
                "y_fin": y_bot,
                "imagen": (
                    f"columna_{col_idx:02d}_cabecera.png"
                    if celda_idx == 0
                    else f"columna_{col_idx:02d}/fila_{celda_idx:02d}.png"
                ),
                "adyacentes_izquierda": []
            }

            # Buscar columna anterior
            pos_actual = columnas_ordenadas.index(col_idx)
            if pos_actual > 0:
                col_prev = columnas_ordenadas[pos_actual - 1]
                filas_prev = celdas_por_columna[col_prev]

                for prev_idx in range(len(filas_prev) - 1):
                    prev_top = filas_prev[prev_idx]
                    prev_bot = filas_prev[prev_idx + 1]

                    # Solapamiento real en pixels
                    solape = min(y_bot, prev_bot) - max(y_top, prev_top)
                    if solape >= min_solapamiento:
                        celda_info["adyacentes_izquierda"].append(prev_idx)

            col_data["celdas"].append(celda_info)

        reporte.append(col_data)

    return reporte


def exportar_json(
    celdas_por_columna: dict[int, list[int]],
    dir_salida: str,
    min_solapamiento: int = 10
) -> tuple[str, list[dict]]:
    """
    Calcula adyacencia y exporta el reporte JSON.
    Retorna (ruta_json, reporte).
    """
    reporte = calcular_adyacencia(celdas_por_columna, min_solapamiento)
    ruta_json = os.path.join(dir_salida, "reporte_celdas.json")

    with open(ruta_json, 'w', encoding='utf-8') as f:
        json.dump(reporte, f, ensure_ascii=False, indent=2)

    return ruta_json, reporte


# ════════════════════════════════════════════
# 7. MODO DEBUG
# ════════════════════════════════════════════

def generar_debug(img: np.ndarray,
                  horizontal: np.ndarray,
                  vertical: np.ndarray,
                  puntos: list[tuple[int, int]],
                  columnas_x: list[int],
                  celdas_por_columna: dict[int, list[int]],
                  reporte: list[dict],
                  dir_salida: str):
    """
    Genera imágenes de depuración:
      - debug_lineas.png      → líneas H y V detectadas (superpuestas)
      - debug_grilla.png      → grilla per-columna sobre la imagen original
      - debug_puntos.png      → puntos de intersección marcados
      - debug_adyacencia.png  → relaciones de adyacencia entre columnas
    """
    os.makedirs(dir_salida, exist_ok=True)
    alto, ancho = img.shape[:2]

    # 1. Líneas detectadas
    lineas_color = img.copy()
    lineas_color[horizontal > 0] = [255, 100, 0]
    lineas_color[vertical > 0] = [0, 100, 255]
    cv2.imwrite(os.path.join(dir_salida, "debug_lineas.png"), lineas_color)

    # 2. Puntos de intersección
    puntos_img = img.copy()
    for (x, y) in puntos:
        cv2.circle(puntos_img, (x, y), 8, (0, 255, 0), -1)
    cv2.imwrite(os.path.join(dir_salida, "debug_puntos.png"), puntos_img)

    # 3. Grilla per-columna
    grilla_img = img.copy()
    for x in columnas_x:
        cv2.line(grilla_img, (x, 0), (x, alto), (200, 0, 0), 2)
    for col_idx, filas_y in celdas_por_columna.items():
        x_izq = columnas_x[col_idx]
        x_der = columnas_x[col_idx + 1]
        for y in filas_y:
            cv2.line(grilla_img, (x_izq, y), (x_der, y), (0, 200, 0), 2)
    primera_y = min(
        (ys[0] for ys in celdas_por_columna.values() if ys),
        default=50
    )
    for col_idx in celdas_por_columna:
        x_centro = (columnas_x[col_idx] + columnas_x[col_idx + 1]) // 2
        cv2.putText(grilla_img, f"C{col_idx}",
                    (x_centro - 10, primera_y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.imwrite(os.path.join(dir_salida, "debug_grilla.png"), grilla_img)

    # 4. Adyacencia: rectangulos de celda con flechas a adyacentes
    adj_img = img.copy()
    # Paleta de colores para distinguir columnas
    paleta = [
        (255, 80, 80), (80, 255, 80), (80, 80, 255),
        (255, 255, 80), (255, 80, 255), (80, 255, 255),
        (200, 128, 50), (128, 50, 200), (50, 200, 128),
        (180, 180, 180),
    ]
    for col_data in reporte:
        col_idx = col_data["columna"]
        color = paleta[col_idx % len(paleta)]
        x1 = columnas_x[col_idx]
        x2 = columnas_x[col_idx + 1]
        for celda in col_data["celdas"]:
            y1 = celda["y_inicio"]
            y2 = celda["y_fin"]
            # Rectángulo de la celda
            cv2.rectangle(adj_img, (x1, y1), (x2, y2), color, 2)
            # Etiqueta
            label = f"C{col_idx}:{celda['celda']}"
            cv2.putText(adj_img, label, (x1 + 4, y1 + 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            # Flechas a adyacentes de la columna anterior
            if col_idx > 0 and celda["adyacentes_izquierda"]:
                col_prev_idx = col_idx - 1
                x_prev = columnas_x[col_prev_idx]
                mid_x_cur = x1
                mid_y_cur = (y1 + y2) // 2
                prev_filas = celdas_por_columna.get(col_prev_idx, [])
                for adj_idx in celda["adyacentes_izquierda"]:
                    if adj_idx + 1 < len(prev_filas):
                        mid_y_prev = (prev_filas[adj_idx] + prev_filas[adj_idx + 1]) // 2
                        mid_x_prev = (x_prev + x1) // 2
                        cv2.arrowedLine(adj_img,
                                        (mid_x_prev, mid_y_prev),
                                        (mid_x_cur, mid_y_cur),
                                        (0, 200, 255), 2, tipLength=0.03)
    cv2.imwrite(os.path.join(dir_salida, "debug_adyacencia.png"), adj_img)

    print(f"  [DEBUG] Imágenes de depuración guardadas en: {dir_salida}/")


# ════════════════════════════════════════════
# 8. PIPELINE PRINCIPAL
# ════════════════════════════════════════════

def procesar(ruta_entrada: str, dir_salida: str, debug: bool = False,
             dpi: int = 300, escala: int = 30, tolerancia: int = 15,
             min_solapamiento: int = 10, margen: int = 0,
             modo_grilla: str = 'interseccion', deskew: bool = True):
    """
    Pipeline completo: cargar → deskew → preprocesar → detectar líneas →
    encontrar intersecciones → construir grilla per-columna → recortar celdas.
    """
    print(f"[1/8] Cargando imagen: {ruta_entrada}")
    img = cargar_imagen(ruta_entrada, dpi=dpi)
    print(f"       Dimensiones: {img.shape[1]}×{img.shape[0]} px")

    if deskew:
        print("[2/8] Detectando y corrigiendo inclinación...")
        img, angulo = corregir_inclinacion(img)
        if abs(angulo) >= 0.01:
            print(f"       Ángulo detectado: {angulo:.2f}° → corregido")
            print(f"       Nuevas dimensiones: {img.shape[1]}×{img.shape[0]} px")
        else:
            print(f"       Ángulo detectado: {angulo:.2f}° → sin corrección necesaria")
    else:
        print("[2/8] Corrección de inclinación desactivada (--no-deskew)")

    print("[3/8] Preprocesando (binarización)...")
    binaria = preprocesar(img)

    print(f"[4/8] Detectando líneas (escala={escala})...")
    horizontal, vertical = detectar_lineas(binaria, escala=escala)

    print("[5/8] Buscando intersecciones...")
    puntos = encontrar_intersecciones(horizontal, vertical)
    print(f"       Intersecciones encontradas: {len(puntos)}")

    if len(puntos) < 4:
        print("[ERROR] Se encontraron menos de 4 intersecciones.")
        print("        No se puede construir una grilla.  Verifica que la")
        print("        imagen contiene una tabla con líneas claras.")
        sys.exit(1)

    print(f"[6/8] Construyendo grilla per-columna (tolerancia={tolerancia}, modo={modo_grilla})...")
    columnas_x, celdas_por_columna = construir_grilla_por_columna(
        puntos, tolerancia=tolerancia, modo=modo_grilla
    )
    print(f"       Columnas detectadas: {len(celdas_por_columna)}")
    for col_idx, filas_y in sorted(celdas_por_columna.items()):
        print(f"         C{col_idx}: {len(filas_y) - 1} celdas")

    print(f"[7/8] Recortando celdas (margen={margen})...")
    stats = recortar_celdas(img, columnas_x, celdas_por_columna, dir_salida,
                            margen=margen)

    print(f"[8/8] Generando reporte JSON (min_solapamiento={min_solapamiento})...")
    ruta_json, reporte = exportar_json(
        celdas_por_columna, dir_salida, min_solapamiento
    )

    if debug:
        generar_debug(img, horizontal, vertical, puntos,
                      columnas_x, celdas_por_columna, reporte, dir_salida)

    print(f"\n[✓] Proceso completado.")
    print(f"    Columnas: {stats['columnas']}")
    print(f"    Celdas:   {stats['celdas_totales']}")
    print(f"    JSON:     {ruta_json}")
    print(f"    Salida:   {os.path.abspath(dir_salida)}")


# ════════════════════════════════════════════
# 9. CLI
# ════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Detecta la grilla de una tabla escaneada y recorta cada celda como imagen.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Ejemplos:
  python cortador_de_rectangulos.py in.pdf
  python cortador_de_rectangulos.py in.pdf --salida ./recortes/ --debug
  python cortador_de_rectangulos.py in.pdf --escala 50 --tolerancia 20
  python cortador_de_rectangulos.py in.pdf --min-solapamiento 5
  python cortador_de_rectangulos.py in.pdf --modo-grilla union
  python cortador_de_rectangulos.py in.pdf --no-deskew
        """
    )
    parser.add_argument("entrada",
                        help="Ruta al archivo de entrada (PDF o imagen PNG/JPG/TIFF)")
    parser.add_argument("--salida", "-s", default="./recortes",
                        help="Carpeta de salida (default: ./recortes)")
    parser.add_argument("--debug", "-d", action="store_true",
                        help="Genera imágenes de depuración")
    parser.add_argument("--dpi", type=int, default=300,
                        help="DPI para conversión PDF→imagen (default: 300)")
    parser.add_argument("--escala", "-e", type=int, default=30,
                        help="Divisor para largo mínimo de línea.  Más alto → "
                             "detecta líneas más cortas (default: 30)")
    parser.add_argument("--tolerancia", "-t", type=int, default=15,
                        help="Pixels de tolerancia para agrupar coordenadas "
                             "y matching de intersecciones (default: 15)")
    parser.add_argument("--min-solapamiento", "-ms", type=int, default=10,
                        help="Pixels mínimos de solapamiento Y para "
                             "considerar adyacencia entre celdas (default: 10)")
    parser.add_argument("--margen", "-m", type=int, default=0,
                        help="Pixels a recortar por dentro de cada celda "
                             "para eliminar bordes de línea (default: 0)")
    parser.add_argument("--modo-grilla", "-mg", default="interseccion",
                        choices=["interseccion", "union"],
                        help="'interseccion': solo Ys en ambos bordes (estricto). "
                             "'union': Ys en al menos un borde (permisivo, "
                             "para tablas con líneas débiles). Default: interseccion")
    parser.add_argument("--no-deskew", action="store_true",
                        help="Desactiva la corrección automática de inclinación")

    args = parser.parse_args()

    if not os.path.isfile(args.entrada):
        print(f"[ERROR] No se encontró el archivo: {args.entrada}")
        sys.exit(1)

    procesar(
        args.entrada, args.salida,
        debug=args.debug,
        dpi=args.dpi,
        escala=args.escala,
        tolerancia=args.tolerancia,
        min_solapamiento=args.min_solapamiento,
        margen=args.margen,
        modo_grilla=args.modo_grilla,
        deskew=not args.no_deskew,
    )


if __name__ == "__main__":
    main()
