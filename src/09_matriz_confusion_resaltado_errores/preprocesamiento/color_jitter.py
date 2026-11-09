"""
Ajuste aleatorio de brillo, contraste, saturación y tono (color jitter).

ADVERTENCIA para este dataset: varias de las enfermedades (tizones,
mildiú, manchas bacterianas) se identifican precisamente por el color
y la textura de la lesión. Un jitter agresivo puede borrar justo la
señal que el modelo necesita aprender — usar con intensidad baja
(valores por defecto) y comparar resultados antes de subirla.
"""

from torchvision import transforms


def construir(brillo=0.2, contraste=0.2, saturacion=0.2, tono=0.05):
    """Devuelve el transform de torchvision (opera sobre imagen PIL)."""
    return transforms.ColorJitter(
        brightness=brillo, contrast=contraste, saturation=saturacion, hue=tono
    )