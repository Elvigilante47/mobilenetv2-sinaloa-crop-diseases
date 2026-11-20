"""
Recorte y zoom aleatorio (random resized crop).

Toma una porción aleatoria de la imagen (entre `escala[0]` y `escala[1]`
del área original) y la reescala al tamaño final. Obliga al modelo a no
depender de que la hoja esté siempre centrada o del mismo tamaño
relativo dentro del encuadre.

NOTA: ya incluye el resize final, así que si se activa reemplaza el
paso de Resize base en vez de sumarse a él (esto se maneja
automáticamente en preprocesamiento/__init__.py).
"""

from torchvision import transforms


def construir(tamano=(224, 224), escala=(0.7, 1.0)):
    """Devuelve el transform de torchvision (opera sobre imagen PIL)."""
    return transforms.RandomResizedCrop(size=tamano, scale=escala)