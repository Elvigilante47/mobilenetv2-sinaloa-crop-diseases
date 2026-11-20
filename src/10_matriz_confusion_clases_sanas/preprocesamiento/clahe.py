"""
CLAHE (Contrast Limited Adaptive Histogram Equalization).

Mejora el contraste local sin distorsionar tanto el color como un
ajuste global de brillo/contraste — buena opción para fotos con
iluminación desigual tomadas en campo.

Requiere opencv-python:
    pip install opencv-python
"""

import numpy as np
from PIL import Image

try:
    import cv2
except ImportError:
    cv2 = None


class Clahe:
    def __init__(self, clip_limit=2.0, tile_grid_size=(8, 8)):
        if cv2 is None:
            raise ImportError(
                "CLAHE requiere opencv-python. Instálalo con: pip install opencv-python"
            )
        self._clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)

    def __call__(self, imagen):
        arreglo = np.array(imagen.convert("RGB"))
        lab = cv2.cvtColor(arreglo, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        l = self._clahe.apply(l)
        lab = cv2.merge((l, a, b))
        resultado = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        return Image.fromarray(resultado)


def construir(clip_limit=2.0, tile_grid_size=(8, 8)):
    """Devuelve el objeto invocable (opera sobre imagen PIL)."""
    return Clahe(clip_limit=clip_limit, tile_grid_size=tile_grid_size)