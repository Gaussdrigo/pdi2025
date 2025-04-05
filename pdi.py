# Importar los módulos necesarios para el manejo de archivos, procesamiento de imágenes y visualización
import os
import cv2
import matplotlib.pyplot as plt

# Definir la ruta completa de la carpeta que contiene las imágenes del dataset.
ruta_del_dataset = r"C:\Users\Admin\Desktop\PDI\Panoramic radiography database"

# Obtener la lista completa de archivos que se encuentran en la carpeta del dataset
lista_de_todos_los_archivos = os.listdir(ruta_del_dataset)

# Inicializar una lista vacía para almacenar únicamente los nombres de archivos que terminan con ".jpg"
lista_de_archivos_jpg = []
for nombre_de_archivo in lista_de_todos_los_archivos:
    # Convertir el nombre del archivo a minúsculas y verificar si termina con la extensión ".jpg"
    if nombre_de_archivo.lower().endswith(".jpg"):
        lista_de_archivos_jpg.append(nombre_de_archivo)

# Mostrar en consola la cantidad total de imágenes encontradas en el dataset
cantidad_de_imagenes = len(lista_de_archivos_jpg)
print("Se han encontrado " + str(cantidad_de_imagenes) + " imágenes en el dataset.")

# Verificar que al menos existe una imagen para continuar con el procesamiento
if cantidad_de_imagenes == 0:
    print("No se encontraron imágenes para procesar.")
    exit(1)

# Seleccionar la primera imagen de la lista para realizar las pruebas de mejora de imagen
nombre_de_la_primera_imagen = lista_de_archivos_jpg[0]
ruta_completa_de_la_primera_imagen = os.path.join(ruta_del_dataset, nombre_de_la_primera_imagen)

# Cargar la imagen original utilizando OpenCV en modo de color (BGR por defecto)
imagen_original = cv2.imread(ruta_completa_de_la_primera_imagen)
if imagen_original is None:
    print("Error al cargar la imagen ubicada en: " + ruta_completa_de_la_primera_imagen)
    exit(1)

# Convertir la imagen original de BGR (formato de OpenCV) a RGB (formato adecuado para matplotlib)
imagen_original_rgb = cv2.cvtColor(imagen_original, cv2.COLOR_BGR2RGB)

# Convertir la imagen original a escala de grises, lo cual es necesario para aplicar las técnicas de ecualización
imagen_en_escala_de_grises = cv2.cvtColor(imagen_original, cv2.COLOR_BGR2GRAY)

# Aplicar la ecualización de histograma tradicional utilizando OpenCV
imagen_ecualizada = cv2.equalizeHist(imagen_en_escala_de_grises)

# Crear un objeto CLAHE (Contrast Limited Adaptive Histogram Equalization) con parámetros por defecto
# El parámetro clipLimit controla el contraste (2.0 es un valor común) y tileGridSize define el tamaño de las subregiones
objeto_CLAHE = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

# Aplicar la técnica CLAHE a la imagen en escala de grises
imagen_CLAHE = objeto_CLAHE.apply(imagen_en_escala_de_grises)

# Visualizar los resultados. Se mostrará la imagen original en color, la imagen en escala de grises, 
# la imagen procesada con la ecualización de histograma y la imagen procesada con CLAHE.

# Configurar una figura de matplotlib con tamaño amplio para mostrar múltiples imágenes
plt.figure(figsize=(16, 8))

# Subgráfico 1: Imagen original en color
plt.subplot(1, 4, 1)
plt.imshow(imagen_original_rgb)
plt.title("Imagen Original (Color)")
plt.axis("off")

# Subgráfico 2: Imagen original en escala de grises
plt.subplot(1, 4, 2)
plt.imshow(imagen_en_escala_de_grises, cmap="gray")
plt.title("Imagen en Escala de Grises")
plt.axis("off")

# Subgráfico 3: Imagen procesada con ecualización de histograma tradicional
plt.subplot(1, 4, 3)
plt.imshow(imagen_ecualizada, cmap="gray")
plt.title("Ecualización de Histograma")
plt.axis("off")

# Subgráfico 4: Imagen procesada con CLAHE
plt.subplot(1, 4, 4)
plt.imshow(imagen_CLAHE, cmap="gray")
plt.title("CLAHE")
plt.axis("off")

# Ajustar el layout para que no se superpongan los subgráficos y mostrar la figura completa
plt.tight_layout()
plt.show()
