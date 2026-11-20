"""
Desenfoque gaussiano (blur).

Simula fotos ligeramente desenfocadas, comunes en captura manual con
celular en campo (pulso, enfoque automático imperfecto, etc.).
"""

from torchvision import transforms


def construir(kernel=(5, 5), sigma=(0.1, 2.0)):
    """Devuelve el transform de torchvision (opera sobre imagen PIL)."""
    return transforms.GaussianBlur(kernel_size=kernel, sigma=sigma)