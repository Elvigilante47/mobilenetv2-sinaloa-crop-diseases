"""
Recorte aleatorio / cutout (random erasing).

Tapa una región rectangular aleatoria de la imagen con ruido o un
color sólido, simulando que parte de la hoja esté oculta por otra
hoja, una sombra, o un insecto encima. Se aplica sobre el TENSOR
(después de ToTensor), como espera torchvision.transforms.RandomErasing.
"""

from torchvision import transforms


def construir(prob=0.5, escala=(0.02, 0.15)):
    """Devuelve el transform de torchvision (opera sobre tensor, tras ToTensor)."""
    return transforms.RandomErasing(p=prob, scale=escala)