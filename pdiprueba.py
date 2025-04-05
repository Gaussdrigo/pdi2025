import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio
from skimage.measure import shannon_entropy

# Definir la ruta completa de la carpeta que contiene las imágenes del dataset
ruta_del_dataset = r"C:\Users\Admin\Desktop\PDI\Panoramic radiography database"

# Obtener la lista completa de archivos que se encuentran en la carpeta del dataset
lista_de_todos_los_archivos = os.listdir(ruta_del_dataset)

# Inicializar una lista vacía para almacenar únicamente los nombres de archivos que terminen con ".jpg"
lista_de_archivos_jpg = []
for nombre_de_archivo in lista_de_todos_los_archivos:
    if nombre_de_archivo.lower().endswith(".jpg"):
        lista_de_archivos_jpg.append(nombre_de_archivo)

# Mostrar en consola la cantidad total de imágenes encontradas en el dataset
cantidad_de_imagenes = len(lista_de_archivos_jpg)
print("Se han encontrado " + str(cantidad_de_imagenes) + " imágenes en el dataset.")

if cantidad_de_imagenes == 0:
    print("No se encontraron imágenes para procesar.")
    exit(1)

# Seleccionar la primera imagen de la lista para realizar las pruebas
nombre_de_la_primera_imagen = lista_de_archivos_jpg[0]
ruta_completa_de_la_primera_imagen = os.path.join(ruta_del_dataset, nombre_de_la_primera_imagen)

# Cargar la imagen original en modo BGR
imagen_original_bgr = cv2.imread(ruta_completa_de_la_primera_imagen)
if imagen_original_bgr is None:
    print("Error al cargar la imagen.")
    exit(1)

# Convertir a RGB para visualizar con matplotlib
imagen_original_rgb = cv2.cvtColor(imagen_original_bgr, cv2.COLOR_BGR2RGB)

# Convertir a escala de grises (para ecualización y métricas)
imagen_gris = cv2.cvtColor(imagen_original_bgr, cv2.COLOR_BGR2GRAY)

# =============================
#  1) Ecualización de Histograma
# =============================
imagen_ecualizada = cv2.equalizeHist(imagen_gris)

# =============================
#  2) CLAHE
# =============================
objeto_CLAHE = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
imagen_CLAHE = objeto_CLAHE.apply(imagen_gris)

# =============================
#   Cálculo de Métricas
# =============================

# 1) AMBE (Absolute Mean Brightness Error)
#    Se calcula la diferencia absoluta entre las medias de brillo de la imagen original y la procesada.
def calcular_AMBE(imagen_original, imagen_mejorada):
    # Asegurarse de que ambas imágenes estén en escala de grises y sean del mismo tipo
    media_original = np.mean(imagen_original)
    media_mejorada = np.mean(imagen_mejorada)
    ambe = abs(media_original - media_mejorada)
    return ambe

# 2) PSNR (Peak Signal-to-Noise Ratio)
#    Usamos la función de skimage.metrics.peak_signal_noise_ratio
#    Asume que ambas imágenes tienen el mismo rango de valores (0-255 en escala de grises).
def calcular_PSNR(imagen_original, imagen_mejorada):
    # Se asume que ambas imágenes son np.uint8 (0 a 255).
    # data_range se puede especificar como 255 si las imágenes están en 8 bits.
    psnr = peak_signal_noise_ratio(imagen_original, imagen_mejorada, data_range=255)
    return psnr

# 3) Contraste (Desviación Estándar)
#    El contraste se estima como la desviación estándar de los valores de la imagen.
def calcular_contraste(imagen):
    # np.std() calcula la desviación estándar de todos los valores en la imagen.
    contraste = np.std(imagen)
    return contraste

# 4) Entropía
#    Se calcula con skimage.measure.shannon_entropy, que asume que la imagen es 2D.
def calcular_entropia(imagen):
    # Se espera que la imagen sea en escala de grises (uint8).
    entropia = shannon_entropy(imagen)
    return entropia

# Calcular las métricas para ecualización de histograma
ambe_ecualizada = calcular_AMBE(imagen_gris, imagen_ecualizada)
psnr_ecualizada = calcular_PSNR(imagen_gris, imagen_ecualizada)
contraste_ecualizada = calcular_contraste(imagen_ecualizada)
entropia_ecualizada = calcular_entropia(imagen_ecualizada)

# Calcular las métricas para CLAHE
ambe_CLAHE = calcular_AMBE(imagen_gris, imagen_CLAHE)
psnr_CLAHE = calcular_PSNR(imagen_gris, imagen_CLAHE)
contraste_CLAHE = calcular_contraste(imagen_CLAHE)
entropia_CLAHE = calcular_entropia(imagen_CLAHE)

# Mostrar los resultados en consola
print("\n--- Resultados Métricas (Ecualización de Histograma) ---")
print(f"AMBE:      {ambe_ecualizada:.3f}")
print(f"PSNR:      {psnr_ecualizada:.3f} dB")
print(f"Contraste: {contraste_ecualizada:.3f}")
print(f"Entropía:  {entropia_ecualizada:.3f}")

print("\n--- Resultados Métricas (CLAHE) ---")
print(f"AMBE:      {ambe_CLAHE:.3f}")
print(f"PSNR:      {psnr_CLAHE:.3f} dB")
print(f"Contraste: {contraste_CLAHE:.3f}")
print(f"Entropía:  {entropia_CLAHE:.3f}")

# Visualización de las imágenes
plt.figure(figsize=(16, 8))

# Subgráfico 1: Imagen Original en color
plt.subplot(1, 4, 1)
plt.imshow(imagen_original_rgb)
plt.title("Imagen Original (Color)")
plt.axis("off")

# Subgráfico 2: Imagen en Escala de Grises
plt.subplot(1, 4, 2)
plt.imshow(imagen_gris, cmap="gray")
plt.title("Imagen en Escala de Grises")
plt.axis("off")

# Subgráfico 3: Ecualización de Histograma
plt.subplot(1, 4, 3)
plt.imshow(imagen_ecualizada, cmap="gray")
plt.title("Ecualización de Histograma")
plt.axis("off")

# Subgráfico 4: CLAHE
plt.subplot(1, 4, 4)
plt.imshow(imagen_CLAHE, cmap="gray")
plt.title("CLAHE")
plt.axis("off")

plt.tight_layout()
plt.show()
