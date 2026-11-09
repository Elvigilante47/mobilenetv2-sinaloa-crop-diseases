"""
Ruido gaussiano aditivo.

Simula sensores de cámara de baja calidad — relevante si la app final
se usará con cámaras de celulares variados en campo. A diferencia de
los demás métodos de esta carpeta, este se aplica sobre el TENSOR
(después de ToTensor), no sobre la imagen PIL, porque opera con
valores numéricos normalizados entre 0 y 1.
"""

import torch


class RuidoGaussiano:
    def __init__(self, media=0.0, desviacion=0.05):
        self.media = media
        self.desviacion = desviacion

    def __call__(self, tensor):
        ruido = torch.randn_like(tensor) * self.desviacion + self.media
        return torch.clamp(tensor + ruido, 0.0, 1.0)


def construir(media=0.0, desviacion=0.05):
    """Devuelve el objeto invocable (opera sobre tensor, tras ToTensor)."""
    return RuidoGaussiano(media=media, desviacion=desviacion)