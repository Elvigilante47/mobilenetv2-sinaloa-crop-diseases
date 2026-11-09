import os
import json
import shutil
from pathlib import Path
from tqdm import tqdm


def cargar_dataset_a_ram(
# el archivo de JSON se a modificado para poder aplicarse al uso de las imagenes redimensionadas 
    json_filename="manifiesto_224.json", 
    ram_base_dir="/dev/shm/dataset_224"
):
    """
    Lee el archivo manifiesto desde 'Tesis/Manifiest/' estando este script 
    ubicado en 'Tesis/Código/', y copia las imágenes a la memoria RAM (/dev/shm).
    """
    # 1. Definir la ruta raíz del proyecto ('Tesis')
    # Path(__file__).resolve().parent es 'Tesis/Código'
    # .parent adicional nos sube un nivel a 'Tesis'
    ROOT_DIR = Path(__file__).resolve().parent.parent
    
    # 2. Construir la ruta al JSON dentro de 'Tesis/Manifiest/'
    manifest_path = ROOT_DIR / "Manifiest" / json_filename
    
    if not manifest_path.exists():
        raise FileNotFoundError(f"No se encontró el manifiesto en: {manifest_path}")

    print(f"Carpeta raíz del proyecto: {ROOT_DIR}")
    print(f"Cargando manifiesto desde: {manifest_path}")
    
    with open(manifest_path, 'r', encoding='utf-8') as f:
        manifest = json.load(f)

    # 3. Resolver la ruta de Horti_Diseases_Dataset
    # Si 'horti_path' es relativa en el JSON, la resolvemos desde 'Tesis'
    horti_path = Path(manifest["horti_path"])
    if not horti_path.is_absolute():
        horti_path = (ROOT_DIR / horti_path).resolve()

    classes_info = manifest.get("classes", {})
    ram_base_path = Path(ram_base_dir)
    
    # 4. Limpiar o crear la carpeta destino en RAM
    if ram_base_path.exists():
        print(f"Limpiando directorio previo en RAM: {ram_base_path}")
        shutil.rmtree(ram_base_path)
    
    ram_base_path.mkdir(parents=True, exist_ok=True)

    print("\nCopiando imágenes a la memoria RAM...")
    
    copiados = 0
    no_encontrados = 0

    # 5. Iterar por cada clase del manifiesto
    for class_name, data in classes_info.items():
        # Crear subcarpeta de la clase en RAM
        target_class_dir = ram_base_path / class_name
        target_class_dir.mkdir(parents=True, exist_ok=True)
        
        # Carpeta origen en el disco duro
        source_class_dir = horti_path / class_name
        selected_files = data.get("selected", [])
        
        print(f"\nProcesando clase '{class_name}' ({len(selected_files)} imágenes)...")
        
        for file_name in tqdm(selected_files, desc=f"Copiando {class_name}"):
            src_file = source_class_dir / file_name
            dst_file = target_class_dir / file_name
            
            if src_file.exists():
                shutil.copy2(src_file, dst_file)
                copiados += 1
            else:
                no_encontrados += 1

    print("\n" + "="*50)
    print("¡Proceso de carga a RAM finalizado exitosamente!")
    print(f"Ruta de los datos en RAM: {ram_base_path.resolve()}")
    print(f"Total de imágenes copiadas: {copiados}")
    if no_encontrados > 0:
        print(f"Advertencia: {no_encontrados} imágenes no se encontraron en la ruta origen.")
    print("="*50)

    return ram_base_path

if __name__ == "__main__":
    JSON_NAME = "manifiesto_224.json"
    ram_dataset_path = cargar_dataset_a_ram(JSON_NAME)