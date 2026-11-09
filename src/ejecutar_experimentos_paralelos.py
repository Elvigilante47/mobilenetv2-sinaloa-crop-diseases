"""
Orquesta el entrenamiento paralelo utilizando 2 GPUs NVIDIA RTX 2080 Ti.
Lanza pares de experimentos simultáneos (uno en GPU 0 y otro en GPU 1).
Asegura la presencia del dataset en RAM antes de iniciar.
"""

from concurrent.futures import ProcessPoolExecutor, as_completed
import os
import sys
import subprocess
import time

# ---------------------------------------------------------------------------
# RUTAS Y CONFIGURACIÓN GENERAL
# ---------------------------------------------------------------------------
CARPETA_ENTRENAMIENTO = (
    "/media/mcc2025/Almacenamiengto/Jared Baldenegro/Tesis/"
    "Codigo/08_entrenamiento_matriz_confusion"
)

# Agregar la carpeta padre 'Codigo' a sys.path para poder importar ram_dataset_loader
CODIGO_DIR = os.path.dirname(CARPETA_ENTRENAMIENTO)
if CODIGO_DIR not in sys.path:
    sys.path.append(CODIGO_DIR)


from ram_dataset_loader_v2 import cargar_dataset_a_ram
#from ram_dataset_loader import cargar_dataset_a_ram

ENTRENAMIENTO_SCRIPT = os.path.join(CARPETA_ENTRENAMIENTO, "entrenamiento.py")
COMPARADOR_SCRIPT = os.path.join(CODIGO_DIR, "comparar_experimentos_paralelos.py")

#CARPETA_REPORTES_DESTINO = "reportes_ram_224_bs64"
#CARPETA_REPORTES_DESTINO = "reportes_ram_224_bs32"
CARPETA_REPORTES_DESTINO = "reportes_prueba01_matriz"
# reportes_ram_224

##RAM_DIR = "/dev/shm/dataset_entrenamiento"
RAM_DIR = "/dev/shm/dataset_224"

REPETICIONES = 4

EXPERIMENTOS = [
    ("baseline", []),
    ("flip_horizontal", ["flip_horizontal"]),
    #("flip_vertical", ["flip_vertical"]),
    #("rotacion", ["rotacion"]),
    #("recorte_zoom", ["recorte_zoom"]),
    ("color_jitter", ["color_jitter"]),
    #("clahe", ["clahe"]),
    ("desenfoque_gaussiano", ["desenfoque_gaussiano"]),
    #("ruido_gaussiano", ["ruido_gaussiano"]),
    ("recorte_aleatorio", ["recorte_aleatorio"]),
]


def verificar_y_preparar_ram(ram_path=RAM_DIR):
    """
    Verifica si la carpeta del dataset en RAM existe y contiene datos.
    Si no existe o está vacía, invoca el cargador a RAM.
    """
    print("🔍 Verificando estado del dataset en RAM (/dev/shm)...")
    
    # Comprobar si existe la carpeta y si tiene subcarpetas con imágenes
    ram_lista = False
    if os.path.exists(ram_path):
        elementos = os.listdir(ram_path)
        if len(elementos) > 0:
            ram_lista = True

    if ram_lista:
        print(f"✅ El dataset en RAM ya está listo en: {ram_path}")
    else:
        print("⚠️ No se encontró el dataset en RAM (o está vacío). Cargando desde disco...")
        cargar_dataset_a_ram(
            #json_filename="manifiesto_5_enfermedades.json",
            json_filename="manifiesto_224.json",
            ram_base_dir=ram_path
        )


def ejecutar_una_corrida(args):
    nombre_exp, metodos_activos, rep, gpu_id = args
    env = os.environ.copy()
    env["PREPROCESAMIENTO_ACTIVO"] = ",".join(metodos_activos)
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    env["REPORTS_DIR_NAME"] = CARPETA_REPORTES_DESTINO

    print(
        f"▶️  [GPU {gpu_id}] Iniciando: {nombre_exp} (Repetición {rep}) en GPU {gpu_id}..."
    )

    inicio = time.time()
    resultado = subprocess.run(
        [sys.executable, ENTRENAMIENTO_SCRIPT],
        cwd=CARPETA_ENTRENAMIENTO,
        env=env,
    )
    duracion = time.time() - inicio

    if resultado.returncode == 0:
        print(
            f"✅ [GPU {gpu_id}] COMPLETADO: {nombre_exp} (Rep {rep}) en {duracion / 60:.1f} min"
        )
        return True, nombre_exp, rep, gpu_id
    else:
        print(f"❌ [GPU {gpu_id}] FALLÓ: {nombre_exp} (Rep {rep})")
        return False, nombre_exp, rep, gpu_id


def main():
    # ⚡ 1. Garantizar la existencia del dataset en RAM
    verificar_y_preparar_ram()

    # 2. Construir lista de tareas
    tareas = []
    for nombre, metodos in EXPERIMENTOS:
        for rep in range(1, REPETICIONES + 1):
            tareas.append((nombre, metodos, rep))

    print(f"\nTotal de corridas a realizar: {len(tareas)}")
    print("Se utilizarán 2 GPUs en paralelo (GPU 0 y GPU 1)")
    print(f"Los reportes se guardarán en: Codigo/{CARPETA_REPORTES_DESTINO}\n" + "=" * 80)

    inicio_total = time.time()

    # 3. Ejecución en paralelo
    with ProcessPoolExecutor(max_workers=2) as executor:
        futures = []
        for i, (nombre, metodos, rep) in enumerate(tareas):
            gpu_id = i % 2  # Alterna entre GPU 0 y GPU 1
            futures.append(
                executor.submit(ejecutar_una_corrida, (nombre, metodos, rep, gpu_id))
            )

        for future in as_completed(futures):
            ok, nombre_exp, rep, gpu_id = future.result()

    duracion_total = time.time() - inicio_total
    print("\n" + "=" * 80)
    print(f"🎉 Barrido completo en paralelo finalizado en: {duracion_total / 3600:.2f} horas")
    print("=" * 80 + "\n")

    # 4. Generar reporte comparativo
    env_comp = os.environ.copy()
    env_comp["REPORTS_DIR_NAME"] = CARPETA_REPORTES_DESTINO
    subprocess.run([sys.executable, COMPARADOR_SCRIPT], env=env_comp)


if __name__ == "__main__":
    main()