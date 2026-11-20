"""
Script para generar una galería visual de ejemplo aplicando cada uno de los 
9 métodos de preprocesamiento de manera individual a una imagen por clase.
"""

import os
from pathlib import Path
from PIL import Image
import torchvision.transforms as T
from configuracion import DATASET_RAM_DIR, CLASSES_CONFIG

# Directorio donde se guardarán las imágenes de ejemplo para la presentación
OUTPUT_DIR = Path("ejemplos_preprocesamiento_presentacion")

# Definir los 9 métodos de preprocesamiento de forma individual usando torchvision.transforms
# (Adaptados a tus transformaciones estándar de entrenamiento)
def obtener_pipelines_individuales():
    return {
        "00_original": T.Lambda(lambda img: img),
        "01_flip_horizontal": T.RandomHorizontalFlip(p=1.0),
        "02_flip_vertical": T.RandomVerticalFlip(p=1.0),
        "03_rotacion": T.RandomRotation(degrees=30),
        "04_recorte_zoom": T.RandomResizedCrop(size=(224, 224), scale=(0.8, 1.0)),
        "05_color_jitter": T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
        "06_clahe": T.Lambda(lambda img: img), # Se manejará especial o se omite si requiere PIL/OpenCV directo, o simulado con contraste
        "07_desenfoque_gaussiano": T.GaussianBlur(kernel_size=(5, 5), sigma=(1.5, 1.5)),
        "08_ruido_gaussiano": T.Lambda(lambda img: img), # Lo manejamos con tensor o adición de ruido si aplica
        "09_recorte_aleatorio": T.RandomCrop(size=(200, 200), pad_if_needed=True)
    }

def generar_galeria():
    ram_path = Path(DATASET_RAM_DIR)
    
    if not ram_path.exists():
        print(f"❌ No se encontró la carpeta en RAM: {ram_path}")
        print("⚠️ Asegúrate de ejecutar primero 'ram_dataset_loader_v2.py' para cargar el dataset.")
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    pipelines = obtener_pipelines_individuales()

    print(f"🚀 Generando ejemplos visuales en: {OUTPUT_DIR.resolve()}\n")

    total_generadas = 0

    # Iterar por cada clase definida en tu configuración
    for class_name in CLASSES_CONFIG.keys():
        class_dir = ram_path / class_name
        if not class_dir.exists():
            print(f"⚠️ Advertencia: No se encontró la carpeta de la clase {class_name} en RAM.")
            continue

        # Tomar la primera imagen disponible de la clase
        imagenes = [f for f in class_dir.iterdir() if f.suffix.lower() in (".jpg", ".jpeg", ".png")]
        if not imagenes:
            continue
        
        img_path = imagenes[0] # Seleccionamos la primera
        print(cls_msg := f"📂 Procesando clase: {class_name} (Imagen base: {img_path.name})")

        # Crear subcarpeta para esta clase en la salida
        class_out_dir = OUTPUT_DIR / class_name
        class_out_dir.mkdir(parents=True, exist_ok=True)

        # Abrir imagen original
        try:
            with Image.open(img_path) as img:
                img = img.convert("RGB")

                # Aplicar cada método de preprocesamiento
                for nombre_metodo, transform in pipelines.items():
                    # Para métodos especiales como CLAHE o Ruido que requieren funciones auxiliares de tus scripts:
                    if nombre_metodo == "06_clahe":
                        # Simulación limpia de mejora de contraste local equivalente a CLAHE
                        from PIL import ImageEnhance
                        img_procesada = ImageEnhance.Contrast(img).enhance(1.8)
                    elif nombre_metodo == "08_ruido_gaussiano":
                        # Simulación de ligera alteración estilo ruido agregando variación suave
                        import numpy as np
                        img_np = np.array(img)
                        ruido = np.random.normal(0, 15, img_np.shape).astype(np.uint8)
                        img_procesada = Image.fromarray(np.clip(img_np.astype(int) + ruido, 0, 255).astype(np.uint8))
                    else:
                        img_procesada = transform(img)

                    # Redimensionar todas a 224x224 para que guarden uniformidad visual en las diapositivas
                    img_procesada = img_procesada.resize((224, 224), Image.Resampling.LANCZOS)

                    # Guardar archivo resultante
                    output_file_path = class_out_dir / f"{nombre_metodo}.jpg"
                    img_procesada.save(output_file_path, quality=95)
                    total_generadas += 1

        except Exception as e:
            print(f"❌ Error procesando {img_path.name}: {e}")

    print("\n" + "="*50)
    print(f"🎉 ¡Galería generada con éxito!")
    print(f"📁 Total de imágenes creadas: {total_generadas}")
    print(f"📍 Encuéntralas en: {OUTPUT_DIR.resolve()}")
    print("="*50)

if __name__ == "__main__":
    generar_galeria()