"""
Script autónomo para generar una copia persistente en disco de todo el dataset
redimensionado a 224x224 (tamaño nativo para MobileNetV2) sin dependencias externas.
"""

import os
from pathlib import Path
from PIL import Image
from tqdm import tqdm

# =============================================================================
# CONFIGURACIÓN DIRECTA Y RUTAS
# =============================================================================
BASE_DRIVE = "/media/mcc2025/Almacenamiengto/Jared Baldenegro/Tesis"

# Ruta origen del dataset completo en el almacenamiento masivo
DATASET_ORIGINAL = os.path.join(BASE_DRIVE, "Datasets", "dataset_local")

# Ruta destino donde se creará la réplica en 224x224
DATASET_NUEVO_224 = os.path.join(BASE_DRIVE, "Datasets", "Dataset_224")

# Tamaño objetivo para MobileNetV2
TARGET_SIZE = (224, 224)

# Extensiones de imagen permitidas
VALID_EXTS = (".jpg", ".jpeg", ".png")


# =============================================================================
# LÓGICA DE PROCESAMIENTO
# =============================================================================
def redimensionar_dataset_completo():
  src_base = Path(DATASET_ORIGINAL)
  dst_base = Path(DATASET_NUEVO_224)

  if not src_base.exists():
    raise FileNotFoundError(
        f"❌ No se encontró el dataset original en la ruta: {src_base}"
    )

  print("=" * 65)
  print("🚀 INICIANDO REDIMENSIONAMIENTO MASIVO DE DATASET")
  print("=" * 65)
  print(f"📁 Origen:  {src_base.resolve()}")
  print(f"📁 Destino: {dst_base.resolve()}")
  print(f"📐 Tamaño:  {TARGET_SIZE}")
  print("=" * 65 + "\n")

  # Obtener todas las subcarpetas (que representan las clases del dataset)
  subcarpetas = [d for d in src_base.iterdir() if d.is_dir()]

  if not subcarpetas:
    print("⚠️ No se encontraron subcarpetas en el directorio origen.")
    return

  total_convertidas = 0
  total_omitidas = 0
  total_errores = 0

  for class_dir in subcarpetas:
    class_name = class_dir.name
    dst_class_dir = dst_base / class_name
    dst_class_dir.mkdir(parents=True, exist_ok=True)

    # Filtrar solo archivos con extensiones válidas de imagen
    archivos = [
        f
        for f in class_dir.iterdir()
        if f.is_file() and f.suffix.lower() in VALID_EXTS
    ]
    print(
        f"\n📂 Clase '{class_name}': {len(archivos)} imágenes encontradas."
    )

    for img_path in tqdm(
        archivos, desc=f"Procesando {class_name}", leave=True
    ):
      dst_img_path = dst_class_dir / img_path.name

      # Si la imagen ya fue procesada anteriormente, se omite para ganar tiempo
      if dst_img_path.exists():
        total_omitidas += 1
        continue

      try:
        with Image.open(img_path) as img:
          # Convertir a RGB (maneja PNGs con transparencia RGBA o fotos en escala de grises)
          img = img.convert("RGB")

          # Redimensionamiento Lanczos (máxima calidad de resampleo)
          img_resized = img.resize(TARGET_SIZE, Image.Resampling.LANCZOS)

          # Guardar la copia optimizada en la carpeta destino
          img_resized.save(dst_img_path, quality=95)

        total_convertidas += 1
      except Exception as e:
        print(f"\n❌ Error al procesar {img_path.name}: {e}")
        total_errores += 1

  print("\n" + "=" * 65)
  print("🎉 ¡PROCESO DE REDIMENSIONAMIENTO A DISCO FINALIZADO!")
  print("=" * 65)
  print(f"✅ Nuevas imágenes convertidas:            {total_convertidas}")
  print(f"⏩ Imágenes omitidas (ya existían):         {total_omitidas}")
  if total_errores > 0:
    print(f"⚠️  Imágenes con error al abrir/guardar:    {total_errores}")
  print(f"📍 Dataset listo en:                       {dst_base.resolve()}")
  print("=" * 65 + "\n")


if __name__ == "__main__":
  redimensionar_dataset_completo()






  