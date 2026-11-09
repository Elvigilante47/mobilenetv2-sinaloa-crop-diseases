"""
Compara todas las corridas de entrenamiento registradas en Codigo/reportes/,
leyendo cada par (*_config.json + *.csv) y armando una tabla resumen — para
no tener que abrir cada archivo por separado cada vez que quieras comparar
resultados de distintos preprocesamientos o arquitecturas.

Uso:
    python comparar_experimentos.py
"""

import os
import glob
import json
import csv

BASE_DRIVE = "/media/mcc2025/Almacenamiengto/Jared Baldenegro/Tesis"
REPORTS_DIR = os.path.join(BASE_DRIVE, "Codigo", "reportes")


def cargar_ultima_epoca(csv_path):
    """Lee el CSV de métricas y devuelve la fila de la última época (no la fila TOTAL)."""
    filas_epocas = []
    duracion = None
    with open(csv_path, newline="") as f:
        for fila in csv.DictReader(f):
            if fila["epoch"] == "TOTAL":
                duracion = fila["duracion_total_seg"]
            else:
                filas_epocas.append(fila)
    if not filas_epocas:
        return None, duracion
    return filas_epocas[-1], duracion


def main():
    config_paths = sorted(glob.glob(os.path.join(REPORTS_DIR, "*_config.json")))

    if not config_paths:
        print(f"No se encontraron corridas registradas en: {REPORTS_DIR}")
        return

    resumen = []
    for config_path in config_paths:
        with open(config_path) as f:
            config = json.load(f)

        run_id = config["run_id"]
        csv_path = os.path.join(REPORTS_DIR, f"{run_id}.csv")
        if not os.path.exists(csv_path):
            print(f"⚠️  No se encontró el CSV para {run_id}, se omite.")
            continue

        ultima_epoca, duracion = cargar_ultima_epoca(csv_path)
        if ultima_epoca is None:
            continue

        metodos_activos = [k for k, v in config["preprocesamiento"].items() if v]

        resumen.append({
            "run_id": run_id,
            "modelo": config["modelo"],
            "preprocesamiento": ", ".join(metodos_activos) if metodos_activos else "ninguno",
            "epochs": config["epochs"],
            "val_acc_final": ultima_epoca["val_acc"],
            "train_acc_final": ultima_epoca["train_acc"],
            "duracion_seg": duracion,
        })

    if not resumen:
        print("No se pudo leer ninguna corrida completa.")
        return

    # Ordena de mayor a menor val_acc para ver de inmediato qué corrida rindió mejor
    resumen.sort(key=lambda fila: float(fila["val_acc_final"]), reverse=True)

    print(f"\n{'RUN_ID':50} {'PREPROCESAMIENTO':30} {'VAL_ACC':>8} {'TRAIN_ACC':>10} {'DUR (s)':>8}")
    print("-" * 112)
    for fila in resumen:
        print(
            f"{fila['run_id']:50} {fila['preprocesamiento']:30} "
            f"{float(fila['val_acc_final']) * 100:7.2f}% {float(fila['train_acc_final']) * 100:9.2f}% "
            f"{fila['duracion_seg']:>8}"
        )

    salida_path = os.path.join(REPORTS_DIR, "comparacion_experimentos.csv")
    with open(salida_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(resumen[0].keys()))
        writer.writeheader()
        writer.writerows(resumen)

    print(f"\n📊 Tabla comparativa guardada en: {salida_path}")


if __name__ == "__main__":
    main()