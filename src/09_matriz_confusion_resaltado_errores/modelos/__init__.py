"""
Registro de arquitecturas disponibles.

Para agregar una nueva arquitectura (EfficientNet, ViT, etc.) y poder
comparar resultados de precisión/exactitud entre ellas:
  1. Crea un archivo nuevo en esta carpeta, ej. `efficientnet.py`, con una
     función `build_model(num_classes)` que devuelva el modelo listo para
     entrenar (pesos preentrenados + capa final ajustada).
  2. Impórtala aquí abajo y agrégala a MODEL_REGISTRY.
  3. Cambia MODEL_NAME en config.py al nombre que le hayas dado.

El resto del código (dataset.py, train.py) no necesita tocarse.
"""

from .mobilenetv2 import build_model as build_mobilenetv2

MODEL_REGISTRY = {
    "mobilenetv2": build_mobilenetv2,
}


def get_model(name, num_classes):
    try:
        builder = MODEL_REGISTRY[name]
    except KeyError:
        raise ValueError(
            f"Arquitectura '{name}' no registrada. Opciones disponibles: "
            f"{list(MODEL_REGISTRY.keys())}"
        )
    return builder(num_classes)