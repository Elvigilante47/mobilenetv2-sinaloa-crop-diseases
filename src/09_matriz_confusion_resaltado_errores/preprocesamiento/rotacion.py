"""
Rotación aleatoria dentro de un rango de grados.

Simula ángulos de cámara distintos al fotografiar hojas en campo,
donde rara vez la hoja queda perfectamente alineada.
"""

from torchvision import transforms


def construir(grados=20):
    """Devuelve el transform de torchvision (opera sobre imagen PIL)."""
    return transforms.RandomRotation(degrees=grados)