"""
Registro de métodos de preprocesamiento/aumento de datos.

Cada método se activa o desactiva desde PREPROCESSING_CONFIG en
configuracion.py, sin tocar este archivo ni dataset.py. Para agregar
un método nuevo:
  1. Crea un archivo aquí con una función `construir(**parametros)` que
     devuelva un transform (o cualquier objeto invocable compatible con
     torchvision.transforms.Compose: recibe una imagen/tensor y
     devuelve una imagen/tensor).
  2. Regístralo abajo en PIPELINE_PIL (si opera sobre la imagen PIL,
     ANTES de ToTensor) o PIPELINE_TENSOR (si opera sobre el tensor,
     DESPUÉS de ToTensor — como ruido_gaussiano o recorte_aleatorio).
  3. Agrega su bandera a PREPROCESSING_CONFIG en configuracion.py.

Nota sobre recorte_zoom: al activarse reemplaza el Resize base en vez
de sumarse a él, porque RandomResizedCrop ya reescala la imagen al
tamaño final por sí mismo (ver dataset.py).
"""

from . import (
    flip_horizontal,
    flip_vertical,
    rotacion,
    recorte_zoom,
    color_jitter,
    clahe,
    desenfoque_gaussiano,
    ruido_gaussiano,
    recorte_aleatorio,
)

# Métodos que operan sobre la imagen PIL, ANTES de ToTensor.
PIPELINE_PIL = {
    "flip_horizontal": flip_horizontal.construir,
    "flip_vertical": flip_vertical.construir,
    "rotacion": rotacion.construir,
    "recorte_zoom": recorte_zoom.construir,
    "color_jitter": color_jitter.construir,
    "clahe": clahe.construir,
    "desenfoque_gaussiano": desenfoque_gaussiano.construir,
}

# Métodos que operan sobre el tensor, DESPUÉS de ToTensor (antes de Normalize).
PIPELINE_TENSOR = {
    "ruido_gaussiano": ruido_gaussiano.construir,
    "recorte_aleatorio": recorte_aleatorio.construir,
}

# Métodos que ya incluyen su propio resize y por lo tanto reemplazan
# el Resize base de dataset.py en vez de sumarse a él.
METODOS_CON_RESIZE_PROPIO = {"recorte_zoom"}


def construir_pipelines(config):
    """
    A partir de PREPROCESSING_CONFIG, arma dos listas de transforms:
    una para aplicar sobre la imagen PIL (antes de ToTensor) y otra
    sobre el tensor (después de ToTensor). Los métodos con bandera
    False simplemente no se incluyen.

    Devuelve: (transforms_pil, transforms_tensor, usa_resize_propio)
    """
    transforms_pil = []
    usa_resize_propio = False

    for nombre, constructor in PIPELINE_PIL.items():
        if config.get(nombre, False):
            transforms_pil.append(constructor())
            if nombre in METODOS_CON_RESIZE_PROPIO:
                usa_resize_propio = True

    transforms_tensor = []
    for nombre, constructor in PIPELINE_TENSOR.items():
        if config.get(nombre, False):
            transforms_tensor.append(constructor())

    return transforms_pil, transforms_tensor, usa_resize_propio