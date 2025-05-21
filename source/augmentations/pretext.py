# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import Tuple
from PIL import ImageOps, ImageFilter
import numpy as np
import torchvision.transforms as transforms
import torch
from timm.data import ToTensor
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD


class GaussianBlur(object):
    def __init__(self, p):
        self.p = p

    def __call__(self, img):
        if np.random.rand() < self.p:
            sigma = np.random.rand() * 1.9 + 0.1
            return img.filter(ImageFilter.GaussianBlur(sigma))  # type: ignore
        else:
            return img


class Solarization(object):
    def __init__(self, p):
        self.p = p

    def __call__(self, img):
        if np.random.rand() < self.p:
            return ImageOps.solarize(img)
        else:
            return img


class ContrastivePretextTransform(torch.nn.Module):
    def __init__(
        self,
        img_size: int = 224,
        interpolation_mode: transforms.InterpolationMode = transforms.InterpolationMode.BICUBIC,
        min_ratio: float = 0.08,
    ):
        super().__init__()
        self.transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    img_size,
                    interpolation=interpolation_mode,
                    antialias=True,
                    ratio=(min_ratio, 1.0),
                ),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomApply(
                    [
                        transforms.ColorJitter(
                            brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1
                        )
                    ],
                    p=0.8,
                ),
                transforms.RandomGrayscale(p=0.2),
                GaussianBlur(p=1.0),
                # Solarization(p=0.0),
                ToTensor(),
                transforms.Normalize(
                    mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD
                ),
            ]
        )
        self.transform_prime = transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    img_size,
                    interpolation=interpolation_mode,
                    antialias=True,
                    ratio=(min_ratio, 1.0),
                ),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomApply(
                    [
                        transforms.ColorJitter(
                            brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1
                        )
                    ],
                    p=0.8,
                ),
                transforms.RandomGrayscale(p=0.2),
                GaussianBlur(p=0.1),
                Solarization(p=0.2),
                ToTensor(),
                transforms.Normalize(
                    mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD
                ),
            ]
        )

    def forward(self, sample) -> Tuple[torch.Tensor, torch.Tensor]:
        x1 = self.transform(sample)
        x2 = self.transform_prime(sample)
        return x1, x2
