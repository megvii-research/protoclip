from torchvision.transforms import Normalize, Compose, RandomResizedCrop, InterpolationMode, ToTensor, Resize, \
    CenterCrop

import numpy as np
import torch
from torch import nn
from torchvision.transforms import transforms

def _convert_to_rgb(image):
    return image.convert('RGB')


def image_transform(
        image_size: int,
        is_train: bool,
        mean=(0.48145466, 0.4578275, 0.40821073),
        std=(0.26862954, 0.26130258, 0.27577711),
        augmentation=None
):
    normalize = Normalize(mean=mean, std=std)
    if is_train:
        if not augmentation:
            return Compose([
                RandomResizedCrop(image_size, scale=(0.9, 1.0), interpolation=InterpolationMode.BICUBIC),
                _convert_to_rgb,
                ToTensor(),
                normalize,
            ])
        elif augmentation == 'protoclip-light-augmentation':
            s = 1
            size = image_size
            color_jitter = transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)
            gaussian_blur = transforms.GaussianBlur(kernel_size=21)
            return Compose([
                transforms.RandomResizedCrop(size=size, scale=(0.5, 1.0), interpolation=InterpolationMode.BICUBIC),
                _convert_to_rgb,
                transforms.RandomHorizontalFlip(),
                transforms.RandomApply([color_jitter], p=0.2),
                transforms.RandomGrayscale(p=0.2),
                transforms.RandomApply([gaussian_blur], p=0.2),
                transforms.ToTensor(),
                normalize
                ])
        
    else:
        return Compose([
            Resize(image_size, interpolation=InterpolationMode.BICUBIC),
            CenterCrop(image_size),
            _convert_to_rgb,
            ToTensor(),
            normalize,
        ])
