"""
Compara las corridas registrando F1-score (o Val Loss/Acc si es anterior) y mejor época obtenida.
"""

import csv
import glob
import json
import os

BASE_DRIVE = "/media/mcc2025/Almacenamiengto/Jared Baldenegro/Tesis"
CARPETA_REPORTES_DESTINO = os.environ.get("REPORTS_DIR_NAME", "reportes_ram_30epocas")
REPORTS_DIR = os.path.join(BASE_DRIVE, "Codigo", CARPETA_REPORTES_DESTINO)


def cargar_datos_csv(csv_path):
  filas_epocas = []
  duracion = None
  with open(csv_path, newline="") as f:
    for fila in csv.DictReader(f):
      if fila["epoch"].startswith("BEST_EPOCH"):
        duracion = fila.get("duracion_total_seg")
      else:
        filas_epocas.append(fila)

  if not filas_epocas:
    return None, duracion

  # Buscar la época que obtuvo el menor VAL_LOSS absoluto
  best_row = min(filas_epocas, key=lambda f: float(f["val_loss"]))
  return best_row, duracion


def main():
  config_paths = sorted(glob.glob(os.path.join(REPORTS_DIR, "*_config.json")))

  if not config_paths:
    print(f"No se encontraron reportes en: {REPORTS_DIR}")
    return

  resumen = []
  for config_path in config_paths:
    with open(config_path) as f:
      config = json.load(f)

    run_id = config["run_id"]
    csv_path = os.path.join(REPORTS_DIR, f"{run_id}.csv")
    if not os.path.exists(csv_path):
      continue

    best_epoch_row, duracion = cargar_datos_csv(csv_path)
    if best_epoch_row is None:
      continue

    metodos_activos = [k for k, v in config["preprocesamiento"].items() if v]

    # Manejo limpio del porcentaje de accuracy
    val_acc_val = float(best_epoch_row.get("val_acc", 0.0))
    val_acc_pct = val_acc_val * 100 if val_acc_val <= 1.0 else val_acc_val

    # Manejo de F1-Macro opcional (por si el CSV viejo no lo incluye)
    f1_val = best_epoch_row.get("val_f1_macro")
    val_f1_macro = float(f1_val) if f1_val is not None else 0.0

    resumen.append({
        "run_id": run_id,
        "modelo": config.get("modelo", "mobilenetv2"),
        "preprocesamiento": (
            ", ".join(metodos_activos) if metodos_activos else "baseline"
        ),
        "best_epoch": best_epoch_row["epoch"],
        "val_loss": float(best_epoch_row["val_loss"]),
        "val_acc_pct": val_acc_pct,
        "val_f1_macro": val_f1_macro,
        "duracion_seg": duracion,
    })

  # Ordenar por F1 si está disponible; si es 0, ordenar por menor Val Loss
  resumen.sort(
      key=lambda x: (
          x["val_f1_macro"] if x["val_f1_macro"] > 0 else -x["val_loss"]
      ),
      reverse=True,
  )

  print(f"\n📁 Reportes leídos desde: {REPORTS_DIR}")
  print(
      f"{'RUN_ID':45} {'PREPROCESAMIENTO':25} {'BEST_EPOCH':>10}"
      f" {'VAL_LOSS':>10} {'VAL_ACC':>10} {'VAL_F1_MACRO':>12}"
  )
  print("-" * 115)

  for r in resumen:
    f1_str = f"{r['val_f1_macro']:12.4f}" if r["val_f1_macro"] > 0 else "N/A"
    print(
        f"{r['run_id']:45} {r['preprocesamiento']:25} {r['best_epoch']:>10}"
        f" {r['val_loss']:10.4f} {r['val_acc_pct']:9.2f}% {f1_str}"
    )

  salida_path = os.path.join(REPORTS_DIR, "comparacion_experimentos.csv")
  with open(salida_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(resumen[0].keys()))
    writer.writeheader()
    writer.writerows(resumen)

  print(f"\n📊 Tabla comparativa guardada en: {salida_path}")


if __name__ == "__main__":
  main()