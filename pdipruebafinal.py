import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio
from skimage.measure import shannon_entropy

# =============================================================================
# Funciones para el cálculo de métricas
# =============================================================================

def calcular_AMBE(imagen_original, imagen_mejorada):
    """
    Calcula el Error Medio Absoluto de Brillo (AMBE) entre la imagen original
    y la imagen procesada. Se basa en la diferencia absoluta entre la media
    de intensidades de ambas imágenes.
    """
    media_original = np.mean(imagen_original)
    media_mejorada = np.mean(imagen_mejorada)
    valor_AMBE = abs(media_original - media_mejorada)
    return valor_AMBE

def calcular_PSNR(imagen_original, imagen_mejorada):
    """
    Calcula la relación pico señal a ruido (PSNR) comparando la imagen original
    y la imagen mejorada. Se asume que las imágenes son de 8 bits (rango 0-255).
    """
    valor_PSNR = peak_signal_noise_ratio(imagen_original, imagen_mejorada, data_range=255)
    return valor_PSNR

def calcular_contraste(imagen):
    """
    Estima el contraste de la imagen como la desviación estándar de sus niveles
    de intensidad.
    """
    valor_contraste = np.std(imagen)
    return valor_contraste

def calcular_entropia(imagen):
    """
    Calcula la entropía de Shannon de la imagen, la cual es una medida de la
    cantidad de información presente en la imagen.
    """
    valor_entropia = shannon_entropy(imagen)
    return valor_entropia

# =============================================================================
# Función para aplicar corrección gamma (tercer algoritmo)
# =============================================================================

def aplicar_correccion_gamma(imagen, valor_gamma=0.5):
    """
    Aplica una corrección gamma a una imagen en escala de grises. 
    Un valor gamma menor a 1 aclara la imagen, mientras que un valor mayor la oscurece.
    Se utiliza una tabla de búsqueda para mejorar el rendimiento.
    """
    # Calcular la inversa de gamma para la transformación
    inversa_gamma = 1.0 / valor_gamma
    # Crear una tabla de búsqueda para mapear cada valor de 0 a 255
    tabla_busqueda = []
    for i in range(256):
        nuevo_valor = ((i / 255.0) ** inversa_gamma) * 255
        tabla_busqueda.append(nuevo_valor)
    # Convertir la tabla a un arreglo de tipo uint8
    tabla_busqueda = np.array(tabla_busqueda, dtype="uint8")
    # Aplicar la tabla de búsqueda a la imagen mediante la función LUT de OpenCV
    imagen_gamma = cv2.LUT(imagen, tabla_busqueda)
    return imagen_gamma

# =============================================================================
# Configuración y carga del dataset
# =============================================================================

# Especificar la ruta del dataset (asegúrate que la ruta sea correcta)
ruta_del_dataset = r"C:\Users\Admin\Desktop\PDI\Panoramic radiography database"

# Verificar si la carpeta existe
if not os.path.exists(ruta_del_dataset):
    print("La ruta especificada para el dataset no existe. Verifica la ruta:", ruta_del_dataset)
    exit(1)

# Obtener todos los nombres de archivos en la carpeta
todos_los_archivos = os.listdir(ruta_del_dataset)

# Crear una lista vacía para almacenar los nombres de archivos que sean imágenes JPG
archivos_jpg = []
for nombre_archivo in todos_los_archivos:
    # Convertir el nombre a minúsculas para una comparación sin distinción de mayúsculas/minúsculas
    nombre_en_minusculas = nombre_archivo.lower()
    if nombre_en_minusculas.endswith(".jpg"):
        archivos_jpg.append(nombre_archivo)

print("Número de imágenes encontradas en el dataset:", len(archivos_jpg))
if len(archivos_jpg) == 0:
    print("No se encontraron imágenes en formato JPG en el dataset.")
    exit(1)

# =============================================================================
# Procesamiento y análisis de todas las imágenes del dataset
# =============================================================================

# Lista para almacenar los resultados de cada imagen
resultados_por_imagen = []

# Crear el objeto CLAHE una vez para reutilizarlo en cada iteración
clahe_objeto = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

# Iterar sobre cada imagen encontrada en el dataset
for archivo in archivos_jpg:
    ruta_imagen_actual = os.path.join(ruta_del_dataset, archivo)
    
    # Cargar la imagen en modo escala de grises
    imagen_gris = cv2.imread(ruta_imagen_actual, cv2.IMREAD_GRAYSCALE)
    if imagen_gris is None:
        print("No se pudo cargar la imagen:", archivo)
        continue

    # Aplicar la ecualización de histograma tradicional
    imagen_ecualizada = cv2.equalizeHist(imagen_gris)
    
    # Aplicar la técnica CLAHE
    imagen_clahe = clahe_objeto.apply(imagen_gris)
    
    # Aplicar la corrección gamma como tercer algoritmo
    imagen_gamma = aplicar_correccion_gamma(imagen_gris, valor_gamma=0.5)
    
    # Calcular las métricas para cada técnica comparándolas con la imagen original en escala de grises
    # Ecualización
    ambe_eq = calcular_AMBE(imagen_gris, imagen_ecualizada)
    psnr_eq = calcular_PSNR(imagen_gris, imagen_ecualizada)
    contraste_eq = calcular_contraste(imagen_ecualizada)
    entropia_eq = calcular_entropia(imagen_ecualizada)
    
    # CLAHE
    ambe_clahe = calcular_AMBE(imagen_gris, imagen_clahe)
    psnr_clahe = calcular_PSNR(imagen_gris, imagen_clahe)
    contraste_clahe = calcular_contraste(imagen_clahe)
    entropia_clahe = calcular_entropia(imagen_clahe)
    
    # Corrección Gamma
    ambe_gamma = calcular_AMBE(imagen_gris, imagen_gamma)
    psnr_gamma = calcular_PSNR(imagen_gris, imagen_gamma)
    contraste_gamma = calcular_contraste(imagen_gamma)
    entropia_gamma = calcular_entropia(imagen_gamma)
    
    # Almacenar los resultados de esta imagen en un diccionario
    resultados_por_imagen.append({
        "Imagen": archivo,
        "AMBE_Ecualizacion": ambe_eq,
        "PSNR_Ecualizacion": psnr_eq,
        "Contraste_Ecualizacion": contraste_eq,
        "Entropia_Ecualizacion": entropia_eq,
        "AMBE_CLAHE": ambe_clahe,
        "PSNR_CLAHE": psnr_clahe,
        "Contraste_CLAHE": contraste_clahe,
        "Entropia_CLAHE": entropia_clahe,
        "AMBE_Gamma": ambe_gamma,
        "PSNR_Gamma": psnr_gamma,
        "Contraste_Gamma": contraste_gamma,
        "Entropia_Gamma": entropia_gamma
    })

# Crear un DataFrame de pandas para facilitar el análisis de resultados
df_resultados = pd.DataFrame(resultados_por_imagen)
print("\nResultados individuales por imagen:")
print(df_resultados.head())

# =============================================================================
# Cálculo de promedios globales de métricas para cada técnica
# =============================================================================

# Preparar un DataFrame con los promedios de cada métrica
datos_promedio = {
    "Metrica": ["AMBE", "PSNR", "Contraste", "Entropia"],
    "Ecualizacion": [
        df_resultados["AMBE_Ecualizacion"].mean(),
        df_resultados["PSNR_Ecualizacion"].mean(),
        df_resultados["Contraste_Ecualizacion"].mean(),
        df_resultados["Entropia_Ecualizacion"].mean()
    ],
    "CLAHE": [
        df_resultados["AMBE_CLAHE"].mean(),
        df_resultados["PSNR_CLAHE"].mean(),
        df_resultados["Contraste_CLAHE"].mean(),
        df_resultados["Entropia_CLAHE"].mean()
    ],
    "Gamma": [
        df_resultados["AMBE_Gamma"].mean(),
        df_resultados["PSNR_Gamma"].mean(),
        df_resultados["Contraste_Gamma"].mean(),
        df_resultados["Entropia_Gamma"].mean()
    ]
}
df_promedios = pd.DataFrame(datos_promedio)
print("\nPromedios globales de métricas:")
print(df_promedios)

# =============================================================================
# Visualización comparativa de promedios mediante gráfico de barras
# =============================================================================

plt.figure(figsize=(10, 6))
indices = np.arange(len(df_promedios["Metrica"]))
ancho_barra = 0.25

plt.bar(indices - ancho_barra, df_promedios["Ecualizacion"], ancho_barra, label="Ecualización")
plt.bar(indices, df_promedios["CLAHE"], ancho_barra, label="CLAHE")
plt.bar(indices + ancho_barra, df_promedios["Gamma"], ancho_barra, label="Corrección Gamma")

plt.xticks(indices, df_promedios["Metrica"])
plt.xlabel("Métrica")
plt.ylabel("Valor Promedio")
plt.title("Comparación de Promedios de Métricas para Técnicas de Mejora de Imagen")
plt.legend()
plt.tight_layout()
plt.show()
