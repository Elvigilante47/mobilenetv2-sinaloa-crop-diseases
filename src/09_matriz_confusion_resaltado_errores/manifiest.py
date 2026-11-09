"""
Generación del manifiesto JSON de train/validation para el experimento de
5 enfermedades.

Ejecutar directamente (`python manifest.py`) para forzar la regeneración,
o simplemente correr train.py, que reutiliza el manifiesto existente si
ya fue generado (ver asegurar_manifiesto()).
"""

import os
import json
import random
from datetime import datetime

from configuracion import (
    DATASET_LOCAL,
    SAVE_JSON_PATH,
    CLASSES_CONFIG,
    SEED,
    N_POR_CLASE,
    N_TRAIN,
    VALID_EXTS,
)

random.seed(SEED)


def generar_manifiesto():
    os.makedirs(os.path.dirname(SAVE_JSON_PATH), exist_ok=True)

    manifest = {
        "seed": SEED,
        "created_at": datetime.now().isoformat(),
        "horti_path": DATASET_LOCAL,
        "n_por_clase": N_POR_CLASE,
        "n_train": N_TRAIN,
        "n_validation": N_POR_CLASE - N_TRAIN,
        "extensiones_validas": list(VALID_EXTS),
        "classes": {},
    }

    for folder_name, norm_name in CLASSES_CONFIG.items():
        class_dir = os.path.join(DATASET_LOCAL, folder_name)
        all_files = [f for f in os.listdir(class_dir) if f.lower().endswith(VALID_EXTS)]
        all_files.sort()

        selected = random.sample(all_files, N_POR_CLASE)

        manifest["classes"][folder_name] = {
            "normalized_name": norm_name,
            "total_available": len(all_files),
            "selected": selected,
            "train": selected[:N_TRAIN],
            "validation": selected[N_TRAIN:],
        }

    with open(SAVE_JSON_PATH, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"Manifiesto generado en: {SAVE_JSON_PATH}")
    return manifest


def asegurar_manifiesto():
    """Genera el manifiesto solo si no existe uno ya guardado en disco."""
    if not os.path.exists(SAVE_JSON_PATH):
        print("No se encontró manifiesto, generando uno nuevo...")
        generar_manifiesto()
    else:
        print(f"Reutilizando manifiesto existente: {SAVE_JSON_PATH}")


if __name__ == "__main__":
    generar_manifiesto()