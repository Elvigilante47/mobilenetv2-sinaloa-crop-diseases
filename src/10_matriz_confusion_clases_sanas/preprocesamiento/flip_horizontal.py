"""
Flip horizontal aleatorio.

Espeja la imagen de izquierda a derecha con una probabilidad dada. Las
hojas no tienen una orientación "correcta" fija en la foto, así que es
una transformación de bajo riesgo y alto beneficio para aumentar el
dataset.
"""

from torchvision import transforms


def construir(prob=0.5):
    """Devuelve el transform de torchvision (opera sobre imagen PIL)."""
    return transforms.RandomHorizontalFlip(p=prob)