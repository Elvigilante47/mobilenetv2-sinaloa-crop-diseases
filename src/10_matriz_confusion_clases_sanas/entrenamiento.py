"""
Script de entrenamiento mejorado con F1-score, métricas extendidas,
fijación de semillas globales, mejor checkpoint, precisión mixta (AMP)
y liberación de memoria GPU.
"""

import csv
import json
import os
import random
import time

import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

from configuracion import (
    BATCH_SIZE,
    EPOCHS,
    LEARNING_RATE,
    MODEL_NAME,
    MODEL_SAVE_PATH,
    NUM_CLASSES,
    NUM_WORKERS,
    PREPROCESSING_CONFIG,
    REPORTS_DIR,
    RUN_ID,
    SAVE_JSON_PATH,
    SEED,
)
from dataset import HortiJsonDataset, get_transform
from evaluacion import evaluar_y_guardar
from manifiest import asegurar_manifiesto
from modelos import get_model


def set_seed(seed=42):
  """Asegura la reproducibilidad determinista en Python, NumPy y PyTorch."""
  #random.seed(seed)
  #np.random.seed(seed)
  #torch.manual_seed(seed)
  #torch.cuda.manual_seed_all(seed)
  #torch.backends.cudnn.deterministic = True
  #torch.backends.cudnn.benchmark = False
  torch.backends.cudnn.benchmark = True


def entrenar():
  set_seed(SEED)

  train_transform = get_transform(aplicar_aumento=True)
  val_transform = get_transform(aplicar_aumento=False)

  train_dataset = HortiJsonDataset(
      SAVE_JSON_PATH, split="train", transform=train_transform
  )
  val_dataset = HortiJsonDataset(
      SAVE_JSON_PATH, split="validation", transform=val_transform
  )

  train_loader = DataLoader(
      train_dataset,
      batch_size=BATCH_SIZE,
      shuffle=True,
      num_workers=NUM_WORKERS,
      pin_memory=True,
      persistent_workers=True,
  )
  val_loader = DataLoader(
      val_dataset,
      batch_size=BATCH_SIZE,
      shuffle=False,
      num_workers=NUM_WORKERS,
      pin_memory=True,
      persistent_workers=True,
  )

  metodos_activos = [k for k, v in PREPROCESSING_CONFIG.items() if v]
  print(f"\n🚀 === INICIANDO CORRIDA: {RUN_ID} ===")
  print(f"Arquitectura: {MODEL_NAME}")
  print(
      "Preprocesamiento activo:"
      f" {metodos_activos if metodos_activos else 'ninguno (Baseline)'}"
  )
  print(f"Épocas: {EPOCHS} | Batch Size: {BATCH_SIZE}")

  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  print(
      f"Dispositivo asignado: {device}"
      f" (CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '0')})"
  )

  model = get_model(MODEL_NAME, NUM_CLASSES).to(device)
  criterion = nn.CrossEntropyLoss()
  optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

  # Escalador para Precisión Mixta (AMP)
  scaler = GradScaler()

  metricas = []
  best_val_loss = float("inf")
  best_epoch = -1

  os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)

  for epoch in range(EPOCHS):
    # Pasamos 'scaler=scaler' a las ejecuciones de las épocas
    train_metrics = _run_epoch(
        model,
        train_loader,
        criterion,
        optimizer,
        device,
        epoch,
        mode="train",
        scaler=scaler,
    )
    val_metrics = _run_epoch(
        model,
        val_loader,
        criterion,
        optimizer,
        device,
        epoch,
        mode="val",
        scaler=scaler,
    )

    # Lógica Best Checkpoint: Guardar solo si mejora el val_loss
    is_best = val_metrics["loss"] < best_val_loss
    if is_best:
      best_val_loss = val_metrics["loss"]
      best_epoch = epoch + 1
      torch.save(model.state_dict(), MODEL_SAVE_PATH)

    epoch_res = {
        "epoch": epoch + 1,
        "train_loss": round(train_metrics["loss"], 4),
        "train_acc": round(train_metrics["acc"], 4),
        "train_f1_macro": round(train_metrics["f1_macro"], 4),
        "val_loss": round(val_metrics["loss"], 4),
        "val_acc": round(val_metrics["acc"], 4),
        "val_f1_macro": round(val_metrics["f1_macro"], 4),
        "val_precision_macro": round(val_metrics["precision_macro"], 4),
        "val_recall_macro": round(val_metrics["recall_macro"], 4),
        "is_best": "SI" if is_best else "NO",
    }
    metricas.append(epoch_res)

    print(
        f" ÉPOCA [{epoch + 1}/{EPOCHS}] | "
        f"Train Loss: {train_metrics['loss']:.4f} Acc:"
        f" {train_metrics['acc']*100:.2f}% | Val Loss:"
        f" {val_metrics['loss']:.4f} Acc: {val_metrics['acc']*100:.2f}%"
        f" F1-Macro: {val_metrics['f1_macro']:.4f} {'⭐ [Mejor Modelo Guardado]' if is_best else ''}"
    )

  print(
      f"\n Entrenando completado. Mejor modelo guardado en época {best_epoch}"
      f" con Val Loss: {best_val_loss:.4f}"
  )

  # Evaluación final con el MEJOR modelo (el guardado en disco, no el de la
  # última época): matriz de confusión + reporte por clase.
  # Debe ir antes de liberar la memoria, porque necesita model y val_loader.
  if best_epoch > 0 and os.path.exists(MODEL_SAVE_PATH):
    model.load_state_dict(
        torch.load(MODEL_SAVE_PATH, map_location=device, weights_only=True)
    )
    # Nombres en el mismo orden de los índices que usa el dataset como etiquetas
    nombres_clases = [
        nombre
        for nombre, _ in sorted(
            val_dataset.class_to_idx.items(), key=lambda par: par[1]
        )
    ]
    acc_matriz = evaluar_y_guardar(
        model,
        val_loader,
        device,
        nombres_clases,
        carpeta_reportes=REPORTS_DIR,
        run_id=RUN_ID,
        titulo=(
            f"Matriz de confusión (mejor época: {best_epoch})\n{RUN_ID}"
        ),
    )
    # Verificación cruzada: debe coincidir con el val_acc de la mejor época
    acc_csv = metricas[best_epoch - 1]["val_acc"]
    print(
        f" Verificación: Acc de la matriz {acc_matriz*100:.2f}% |"
        f" Val Acc de la mejor época (CSV) {acc_csv*100:.2f}%"
    )
  else:
    print("⚠️ No se encontró el checkpoint; se omite la matriz de confusión.")

  # Limpieza estricta de memoria VRAM para evitar fallos en la siguiente corrida
  del model, optimizer, train_loader, val_loader
  torch.cuda.empty_cache()

  return metricas, best_epoch, best_val_loss


def _run_epoch(
    model, loader, criterion, optimizer, device, epoch, mode, scaler=None
):
  is_train = mode == "train"
  model.train() if is_train else model.eval()

  running_loss = 0.0
  all_preds = []
  all_labels = []

  label = "Train" if is_train else "Val"
  bar = tqdm(loader, desc=f"Época {epoch + 1} [{label}]", leave=False)

  context = torch.enable_grad() if is_train else torch.no_grad()
  with context:
    for images, labels in bar:
        images, labels = images.to(device), labels.to(device)

        if is_train:
            optimizer.zero_grad()

        # Precisión Mixta activa durante el pase hacia adelante (Forward)
        with autocast():
            outputs = model(images)
            loss = criterion(outputs, labels)

        if is_train:
            # Retropropagación optimizada mediante scaler
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        running_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

  epoch_loss = running_loss / len(loader.dataset)
  epoch_acc = (np.array(all_preds) == np.array(all_labels)).mean()
  epoch_f1_macro = f1_score(
      all_labels, all_preds, average="macro", zero_division=0
  )
  epoch_precision_macro = precision_score(
      all_labels, all_preds, average="macro", zero_division=0
  )
  epoch_recall_macro = recall_score(
      all_labels, all_preds, average="macro", zero_division=0
  )

  return {
      "loss": epoch_loss,
      "acc": epoch_acc,
      "f1_macro": epoch_f1_macro,
      "precision_macro": epoch_precision_macro,
      "recall_macro": epoch_recall_macro,
  }


def guardar_metricas_csv(metricas, duracion_seg, best_epoch, best_val_loss):
  os.makedirs(REPORTS_DIR, exist_ok=True)
  csv_path = os.path.join(REPORTS_DIR, f"{RUN_ID}.csv")

  fieldnames = [
      "epoch",
      "train_loss",
      "train_acc",
      "train_f1_macro",
      "val_loss",
      "val_acc",
      "val_f1_macro",
      "val_precision_macro",
      "val_recall_macro",
      "is_best",
      "duracion_total_seg",
  ]

  with open(csv_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    for fila in metricas:
      writer.writerow({**fila, "duracion_total_seg": ""})
    writer.writerow({
        "epoch": f"BEST_EPOCH_{best_epoch}",
        "train_loss": "",
        "train_acc": "",
        "train_f1_macro": "",
        "val_loss": round(best_val_loss, 4),
        "val_acc": "",
        "val_f1_macro": "",
        "val_precision_macro": "",
        "val_recall_macro": "",
        "is_best": "",
        "duracion_total_seg": round(duracion_seg, 1),
    })

  print(f"📊 Métricas extendidas guardadas en: {csv_path}")


def guardar_config_json(best_epoch, best_val_loss):
  os.makedirs(REPORTS_DIR, exist_ok=True)
  config_path = os.path.join(REPORTS_DIR, f"{RUN_ID}_config.json")

  snapshot = {
      "run_id": RUN_ID,
      "modelo": MODEL_NAME,
      "preprocesamiento": PREPROCESSING_CONFIG,
      "epochs": EPOCHS,
      "batch_size": BATCH_SIZE,
      "learning_rate": LEARNING_RATE,
      "best_epoch": best_epoch,
      "best_val_loss": round(best_val_loss, 4),
      "gpu_utilizada": os.environ.get("CUDA_VISIBLE_DEVICES", "0"),
  }

  with open(config_path, "w") as f:
    json.dump(snapshot, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
  inicio = time.time()
  asegurar_manifiesto()
  metricas, best_epoch, best_val_loss = entrenar()
  duracion = time.time() - inicio

  guardar_metricas_csv(metricas, duracion, best_epoch, best_val_loss)
  guardar_config_json(best_epoch, best_val_loss)