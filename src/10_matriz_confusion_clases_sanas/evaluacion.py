"""
Evaluación final del mejor modelo: matriz de confusión (PNG) y reporte por
clase (JSON con precision, recall y F1).

Se invoca una sola vez, al terminar el entrenamiento, con los pesos del mejor
checkpoint ya cargados en el modelo (ver entrenamiento.py).
"""

import json
import os

import matplotlib

# Backend sin pantalla: este código corre en segundo plano (nohup/subprocess)
# y solo guarda archivos, nunca abre ventanas. Debe fijarse ANTES de importar
# pyplot, de lo contrario puede fallar en procesos sin entorno gráfico.
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.metrics import classification_report, confusion_matrix


def obtener_predicciones(model, loader, device):
  """Recorre el loader una vez y devuelve (y_true, y_pred) como arreglos."""
  model.eval()
  y_true, y_pred = [], []

  with torch.no_grad():
    for images, labels in loader:
      images = images.to(device)
      # Misma precisión mixta que en la validación del entrenamiento, para que
      # las predicciones coincidan con las métricas ya registradas en el CSV.
      with torch.autocast(device_type="cuda", enabled=(device.type == "cuda")):
        outputs = model(images)
      y_pred.extend(outputs.argmax(dim=1).cpu().numpy())
      y_true.extend(labels.numpy())

  return np.array(y_true), np.array(y_pred)


def guardar_matriz_confusion(
    y_true, y_pred, nombres_clases, ruta_png, titulo="Matriz de confusión"
):
  """Genera y guarda la matriz de confusión en formato PNG."""
  # labels=... fuerza una matriz NxN aunque alguna clase no aparezca en
  # y_true/y_pred (evita matrices de tamaño inconsistente entre corridas).
  indices = list(range(len(nombres_clases)))
  cm = confusion_matrix(y_true, y_pred, labels=indices)

  plt.figure(figsize=(10, 8))
  sns.heatmap(
      cm,
      annot=True,
      fmt="d",
      cmap="Blues",
      xticklabels=nombres_clases,
      yticklabels=nombres_clases,
  )
  plt.title(titulo, fontsize=10)
  plt.ylabel("Clase real")
  plt.xlabel("Clase predicha")
  plt.xticks(rotation=45, ha="right")
  plt.yticks(rotation=0)
  plt.tight_layout()
  plt.savefig(ruta_png, dpi=300)
  plt.close()


def generar_reporte_detallado(y_true, y_pred, nombres_clases, ruta_json):
  """Calcula y guarda el reporte por clase (precision, recall, F1) en JSON."""
  indices = list(range(len(nombres_clases)))
  reporte = classification_report(
      y_true,
      y_pred,
      labels=indices,
      target_names=nombres_clases,
      output_dict=True,
      zero_division=0,
  )

  with open(ruta_json, "w", encoding="utf-8") as f:
    # default=float: red de seguridad por si algún valor llega como tipo NumPy.
    json.dump(reporte, f, indent=2, ensure_ascii=False, default=float)

  return reporte


def evaluar_y_guardar(
    model,
    loader,
    device,
    nombres_clases,
    carpeta_reportes,
    run_id,
    titulo=None,
):
  """
  Punto de entrada único: predice sobre `loader`, guarda la matriz de
  confusión y el reporte por clase, y devuelve la accuracy obtenida.

  Los archivos llevan el RUN_ID en el nombre (igual que el .pth, el .csv y el
  _config.json), así cada corrida de un barrido genera los suyos sin
  sobrescribir los de otra corrida que comparte la misma carpeta de reportes.
  """
  os.makedirs(carpeta_reportes, exist_ok=True)

  y_true, y_pred = obtener_predicciones(model, loader, device)

  ruta_png = os.path.join(carpeta_reportes, f"{run_id}_matriz_confusion.png")
  ruta_json = os.path.join(carpeta_reportes, f"{run_id}_reporte_clases.json")

  guardar_matriz_confusion(
      y_true,
      y_pred,
      nombres_clases,
      ruta_png,
      titulo=titulo or f"Matriz de confusión — {run_id}",
  )
  generar_reporte_detallado(y_true, y_pred, nombres_clases, ruta_json)

  print(f"🧩 Matriz de confusión guardada en: {ruta_png}")
  print(f"📋 Reporte por clase guardado en:   {ruta_json}")

  return float((y_true == y_pred).mean())