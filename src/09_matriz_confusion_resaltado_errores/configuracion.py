"""
Configuración centralizada para el entrenamiento en RAM (AgroVision).
"""

from datetime import datetime
import os

# ---------------------------------------------------------------------------
# ARQUITECTURA Y PARÁMETROS GENERALES
# ---------------------------------------------------------------------------
MODEL_NAME = "mobilenetv2"
EPOCHS = 30
#BATCH_SIZE = 128
#BATCH_SIZE = 64
BATCH_SIZE = 32
NUM_WORKERS = 4  # Al estar en RAM, los workers leen casi sin latencia de I/O
#LEARNING_RATE = 2e-4 # Para uso con Batch Size 128
LEARNING_RATE = 1e-4 # Para uso con Batch Size 64 y 32

# ---------------------------------------------------------------------------
# RUTAS — AJUSTA A TU ENTORNO
# ---------------------------------------------------------------------------
BASE_DRIVE = "/media/mcc2025/Almacenamiengto/Jared Baldenegro/Tesis"

#El siguiente es el dataset con las imagenes en tamaño original:
#DATASET_LOCAL = os.path.join(BASE_DRIVE, "Datasets", "dataset_local")

#Este es el dataset con las imagenes redimensionadas (224x224):
DATASET_LOCAL = os.path.join(BASE_DRIVE, "Datasets", "Dataset_224")

# 🚀 Ruta por defecto a la RAM
#DATASET_RAM_DIR = os.environ.get("DATASET_RAM_DIR", "/dev/shm/dataset_entrenamiento")


# Cambia el valor por defecto o la variable de entorno
DATASET_RAM_DIR = os.environ.get("DATASET_RAM_DIR", "/dev/shm/dataset_224")

SAVE_JSON_PATH = os.path.join(
    BASE_DRIVE, "Manifiest", "manifiesto_5_enfermedades.json"
)

# Carpeta personalizada para reportes en RAM
NOMBRE_CARPETA_REPORTES = os.environ.get("REPORTS_DIR_NAME", "reportes_ram_30epocas")
REPORTS_DIR = os.path.join(BASE_DRIVE, "Codigo", NOMBRE_CARPETA_REPORTES)

# ---------------------------------------------------------------------------
# MÉTODOS DE PREPROCESAMIENTO / AUMENTO DE DATOS
# ---------------------------------------------------------------------------
_METODOS_DISPONIBLES = [
    "flip_horizontal",
    "flip_vertical",
    "rotacion",
    "recorte_zoom",
    "color_jitter",
    "clahe",
    "desenfoque_gaussiano",
    "ruido_gaussiano",
    "recorte_aleatorio",
]

_override = os.environ.get("PREPROCESAMIENTO_ACTIVO")
if _override is not None:
    _activos = {m.strip() for m in _override.split(",") if m.strip()}
    PREPROCESSING_CONFIG = {
        metodo: (metodo in _activos) for metodo in _METODOS_DISPONIBLES
    }
else:
    PREPROCESSING_CONFIG = {metodo: False for metodo in _METODOS_DISPONIBLES}

# ---------------------------------------------------------------------------
# IDENTIFICADOR ÚNICO DE ESTA CORRIDA
# ---------------------------------------------------------------------------
_metodos_activos_lista = sorted(k for k, v in PREPROCESSING_CONFIG.items() if v)
_tag_preprocesamiento = (
    "_".join(_metodos_activos_lista)
    if _metodos_activos_lista
    else "sin_preprocesamiento"
)

_gpu_id = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
RUN_ID = f"{MODEL_NAME}_5clases_RAM_{_tag_preprocesamiento}_{EPOCHS}ep_{datetime.now().strftime('%Y%m%d_%H%M%S')}_gpu{_gpu_id}"

MODEL_SAVE_PATH = os.path.join(BASE_DRIVE, "Modelos", f"{RUN_ID}.pth")

# ---------------------------------------------------------------------------
# CLASES Y SPLIT
# ---------------------------------------------------------------------------
CLASSES_CONFIG = {
    "Cucumber_gummy_stem_blight": "cucumber_gummy_stem_blight",
    "Pumpkin_leaf_blight": "pumpkin_leaf_blight",
    "Basil_downy_mildew": "basil_downy_mildew",
    "Melon_gummy_stem_blight": "melon_gummy_stem_blight",
    "Chiltepin_chili_early_Bacterial_blight": (
        "chiltepin_chili_early_bacterial_blight"
    ),
}

NUM_CLASSES = len(CLASSES_CONFIG)
SEED = 42
N_POR_CLASE = 900
N_TRAIN = 720
VALID_EXTS = (".jpg", ".jpeg", ".png")

IMG_SIZE = (224, 224)
NORM_MEAN = [0.485, 0.456, 0.406]
NORM_STD = [0.229, 0.224, 0.225]