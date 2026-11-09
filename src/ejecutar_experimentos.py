"""
Orquesta múltiples corridas de entrenamiento, una por cada configuración
definida en EXPERIMENTOS, repitiendo cada una REPETICIONES veces. Cada
corrida se lanza como un proceso de Python independiente (subprocess),
así que ninguna corrida hereda estado de la anterior.

Al terminar todas las corridas, ejecuta automáticamente
comparar_experimentos.py para generar la tabla resumen final.

AJUSTA antes de correr:
  - CARPETA_ENTRENAMIENTO: la carpeta donde vive tu entrenamiento.py
    actual (la que ya tiene soporte de preprocesamiento).
  - EXPERIMENTOS y REPETICIONES, según lo que quieras probar.

Uso:
    python ejecutar_experimentos.py

Recomendación: como esto puede tardar varias horas, si trabajas por
TeamViewer/StreamViewer en la máquina del laboratorio, córrelo dentro de
`tmux` o `screen` (o con `nohup python ejecutar_experimentos.py &`), para
que el barrido siga corriendo aunque se corte la conexión remota.
"""

import os
import subprocess
import sys
import time

# ---------------------------------------------------------------------------
# AJUSTA ESTO A TU MÁQUINA
# ---------------------------------------------------------------------------
CARPETA_ENTRENAMIENTO = (
    "/media/mcc2025/Almacenamiengto/Jared Baldenegro/Tesis/"
    "Codigo/03 - entrenamiento preprocesamiento automatico"
)
ENTRENAMIENTO_SCRIPT = os.path.join(CARPETA_ENTRENAMIENTO, "entrenamiento.py")
COMPARADOR_SCRIPT = os.path.join(
    "/media/mcc2025/Almacenamiengto/Jared Baldenegro/Tesis/Codigo",
    "comparar_experimentos.py",
)

REPETICIONES = 3

# Cada entrada: (nombre_para_mostrar, lista_de_métodos_activos)
# Lista vacía = baseline, sin ningún método activo.
EXPERIMENTOS = [
    ("baseline", []),
    ("flip_horizontal", ["flip_horizontal"]),
    ("flip_vertical", ["flip_vertical"]),
    ("rotacion", ["rotacion"]),
    ("recorte_zoom", ["recorte_zoom"]),
    ("color_jitter", ["color_jitter"]),
    ("clahe", ["clahe"]),
    ("desenfoque_gaussiano", ["desenfoque_gaussiano"]),
    ("ruido_gaussiano", ["ruido_gaussiano"]),
    ("recorte_aleatorio", ["recorte_aleatorio"]),
]


def ejecutar_una_corrida(metodos_activos):
    env = os.environ.copy()
    env["PREPROCESAMIENTO_ACTIVO"] = ",".join(metodos_activos)

    resultado = subprocess.run(
        [sys.executable, ENTRENAMIENTO_SCRIPT],
        cwd=CARPETA_ENTRENAMIENTO,
        env=env,
    )
    return resultado.returncode == 0


def main():
    total_corridas = len(EXPERIMENTOS) * REPETICIONES
    corrida_actual = 0
    fallidas = []

    inicio_total = time.time()

    for nombre, metodos in EXPERIMENTOS:
        for rep in range(1, REPETICIONES + 1):
            corrida_actual += 1
            print(
                f"\n{'=' * 80}\n"
                f"Corrida {corrida_actual}/{total_corridas} — "
                f"{nombre} (repetición {rep}/{REPETICIONES})\n"
                f"{'=' * 80}"
            )
            inicio = time.time()
            ok = ejecutar_una_corrida(metodos)
            duracion = time.time() - inicio

            if ok:
                print(f"✅ Corrida completada en {duracion / 60:.1f} min")
            else:
                print("❌ La corrida falló (ver traceback arriba)")
                fallidas.append(f"{nombre} (repetición {rep})")

    duracion_total = time.time() - inicio_total
    print(f"\n{'=' * 80}")
    print(f"Barrido completo: {total_corridas} corridas en {duracion_total / 3600:.2f} horas")
    if fallidas:
        print(f"⚠️  Corridas que fallaron: {fallidas}")
    print(f"{'=' * 80}\n")

    print("Generando tabla comparativa...")
    subprocess.run([sys.executable, COMPARADOR_SCRIPT])


if __name__ == "__main__":
    main()