import torch
import torchvision.transforms as transforms
from timm.data import ToTensor
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD


class DownstreamClassificationTrainTransform(torch.nn.Module):
    def __init__(
        self,
        img_size: int = 224,
        min_ratio: float = 0.08,
    ):
        super().__init__()
        self.transform_downstream = transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    img_size,
                    ratio=(min_ratio, 1.0),
                ),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD
                ),
            ]
        )

    def forward(self, sample):
        return self.transform_downstream(sample)


class DownstreamClassificaitonTestTransform(torch.nn.Module):
    def __init__(self, img_size: int = 224):
        super().__init__()
        imagenet_crop_ratio = 224 / 256

        self.transform = transforms.Compose(
            [
                transforms.Resize(
                    int(img_size / imagenet_crop_ratio),
                ),
                transforms.CenterCrop(img_size),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD
                ),
            ]
        )

    def forward(self, sample):
        return self.transform(sample)


class ToTensorMultiInputs(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.to_tensor = ToTensor()

    def forward(self, image, targets):
        return self.to_tensor(image), targets
