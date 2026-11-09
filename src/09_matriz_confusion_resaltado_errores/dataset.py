"""
Dataset de PyTorch con fallback dinámico a RAM o Disco Mecánico.
"""

import os
import json

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from configuracion import (
    IMG_SIZE, 
    NORM_MEAN, 
    NORM_STD, 
    PREPROCESSING_CONFIG,
    DATASET_RAM_DIR
)
from preprocesamiento import construir_pipelines


def get_transform(aplicar_aumento=False):
    if aplicar_aumento:
        transforms_pil, transforms_tensor, usa_resize_propio = construir_pipelines(
            PREPROCESSING_CONFIG
        )
    else:
        transforms_pil, transforms_tensor, usa_resize_propio = [], [], False

    pasos = []
    #if not usa_resize_propio:
    #    pasos.append(transforms.Resize(IMG_SIZE))

    pasos.extend(transforms_pil)
    pasos.append(transforms.ToTensor())
    pasos.extend(transforms_tensor)
    pasos.append(transforms.Normalize(mean=NORM_MEAN, std=NORM_STD))

    return transforms.Compose(pasos)


class HortiJsonDataset(Dataset):
    def __init__(self, json_path, split="train", transform=None, ram_dir=DATASET_RAM_DIR):
        with open(json_path, "r") as f:
            self.data = json.load(f)

        # 🚀 Validación de lectura rápida desde RAM
        if ram_dir and os.path.exists(ram_dir):
            self.horti_path = ram_dir
        else:
            self.horti_path = self.data["horti_path"]

        self.transform = transform
        self.samples = []

        self.class_to_idx = {
            class_name: idx for idx, class_name in enumerate(self.data["classes"].keys())
        }

        for class_name, content in self.data["classes"].items():
            file_list = content[split]
            label = self.class_to_idx[class_name]
            for fname in file_list:
                img_path = os.path.join(self.horti_path, class_name, fname)
                self.samples.append((img_path, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label