"""
Flip vertical aleatorio.

Espeja la imagen de arriba a abajo. Menos natural que el flip
horizontal para fotos de hojas tomadas normalmente, pero puede
ayudar si tu dataset incluye ángulos de cámara variados.
"""

from torchvision import transforms


def construir(prob=0.5):
    """Devuelve el transform de torchvision (opera sobre imagen PIL)."""
    return transforms.RandomVerticalFlip(p=prob)