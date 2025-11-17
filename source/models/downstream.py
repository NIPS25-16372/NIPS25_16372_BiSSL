import copy
import warnings
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple, get_args

import torch
from torch import nn
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.ops import MultiScaleRoIAlign

from source.types import BackboneArchs, ConvBackbones, DetectionBackbones

from .misc import ConvBackbone


### CLASSIFICAITON ###
class DSClassifier(nn.Module):
    def __init__(
        self,
        backbone_arch: BackboneArchs,
        n_classes: int = 10,
    ):
        super().__init__()
        if backbone_arch in get_args(ConvBackbones):
            self.backbone, self.num_features = ConvBackbone(backbone_arch)  # type: ignore
            assert self.num_features is not None
            self.head = nn.Linear(self.num_features, n_classes)
        else:
            raise ValueError(f"Backbone {backbone_arch} not supported")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(x))
