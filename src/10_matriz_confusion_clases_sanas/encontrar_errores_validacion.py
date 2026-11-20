"""
Script de diagnóstico avanzado para identificar las imágenes exactas de validación
que fueron mal clasificadas por un modelo específico y guardarlas en un .txt.
"""

import os
import glob
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from dataset import HortiJsonDataset, get_transform
from configuracion import (
    SAVE_JSON_PATH,
    NUM_CLASSES,
    NUM_WORKERS,
    BASE_DRIVE
)
from modelos import get_model
from configuracion import MODEL_NAME

def analizar_errores():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🔍 Usando dispositivo: {device}")

    # =========================================================================
    # SELECCIÓN EXPLÍCITA DEL MODELO A ANALIZAR
    # =========================================================================
    # Escribe aquí el RUN_ID exacto de la corrida que deseas revisar (sin el .pth),
    # o déjalo en None para que tome automáticamente el último modelo guardado.
    RUN_ID_ESPECIFICO = "mobilenetv2_5clases_RAM_desenfoque_gaussiano_30ep_20261008_135949_gpu0"
    
    # "mobilenetv2_5clases_RAM_flip_horizontal_30ep_20261008_121405_gpu0"
    
    CARPETA_MODELOS = os.path.join(BASE_DRIVE, "Modelos")

    if RUN_ID_ESPECIFICO:
        model_path = os.path.join(CARPETA_MODELOS, f"{RUN_ID_ESPECIFICO}.pth")
    else:
        # Búsqueda automática del archivo .pth más reciente
        archivos_pth = glob.glob(os.path.join(CARPETA_MODELOS, "*.pth"))
        if not archivos_pth:
            print(f"❌ No se encontraron archivos .pth en {CARPETA_MODELOS}")
            return
        model_path = max(archivos_pth, key=os.path.getmtime)

    print(f"📂 Modelo seleccionado para análisis: {model_path}")

    # 1. Cargar transformaciones de validación (sin aumentos)
    val_transform = get_transform(aplicar_aumento=False)

    # 2. Cargar el dataset de validación usando el manifiesto y dataset existentes
    val_dataset = HortiJsonDataset(
        SAVE_JSON_PATH, split="validation", transform=val_transform
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True
    )

    # 3. Cargar el modelo
    if not os.path.exists(model_path):
        print(f"❌ No se encontró el archivo del modelo en: {model_path}")
        return

    model = get_model(MODEL_NAME, NUM_CLASSES).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()

    # Obtener nombres de clases ordenados por índice
    nombres_clases = [
        nombre for nombre, _ in sorted(
            val_dataset.class_to_idx.items(), key=lambda par: par[1]
        )
    ]

    print("\n" + "="*60)
    print("🔍 BUSCANDO IMÁGENES MAL CLASIFICADAS EN VALIDACIÓN")
    print("="*60)

    errores_encontrados = 0
    idx_global = 0

    # Abrir el archivo .txt para registrar los errores automáticamente
    with open("errores_validacion.txt", "w", encoding="utf-8") as f_txt:
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, preds = torch.max(outputs, 1)

                for i in range(images.size(0)):
                    path_imagen, true_label_idx = val_dataset.samples[idx_global]
                    pred_label_idx = preds[i].item()

                    if pred_label_idx != true_label_idx:
                        errores_encontrados += 1
                        clase_real = nombres_clases[true_label_idx]
                        clase_predicha = nombres_clases[pred_label_idx]

                        # Imprimir en consola
                        print(f"\n❌ Error #{errores_encontrados}")
                        print(f"    Ruta del archivo: {path_imagen}")
                        print(f"    Clase Real:     {clase_real}")
                        print(f"    Predicción IA:  {clase_predicha}")
                        print("-" * 50)

                        # Guardar línea en el archivo .txt
                        f_txt.write(f"Archivo: {path_imagen} | Real: {clase_real} | Predicción: {clase_predicha}\n")

                    idx_global += 1

    print(f"\n✨ Análisis finalizado. Total de errores detectados: {errores_encontrados}")
    print(f"📄 Los errores han sido guardados en el archivo: errores_validacion.txt")

if __name__ == "__main__":
    analizar_errores()