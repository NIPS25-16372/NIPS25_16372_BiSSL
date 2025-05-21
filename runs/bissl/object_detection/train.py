from typing import List, Optional, Tuple, Dict
import torch

from source.models.downstream import FasterRCNNR50C4BB
from source.models.simclr import SimCLR
from source.models.byol import BYOL
from source.bissl.bissl_trainer import GeneralBiSSLTrainer


class ObjectDetectionBiSSLTrainer(GeneralBiSSLTrainer):
    """
    Trainer for downstream object detection tasks using BiSSL.

    This class extends GeneralBiSSLTrainer and is specifically designed for
    downstream object detecton tasks. It expects the model to be an
    instance of FasterRCNNExtBB.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        assert isinstance(self.models["upper"].module, FasterRCNNR50C4BB)

        # Object detection is currently only compatible with SimCLR and BYOL
        assert isinstance(self.models["lower"].module, (SimCLR, BYOL))

    def upper_criterion_ext_backbone(
        self,
        upper_input: Tuple[List[torch.Tensor], List[Dict[str, torch.Tensor]]],
        ext_backbone: Optional[torch.nn.Module] = None,
    ) -> torch.Tensor:
        imgs, targets = upper_input
        loss_dict = self.models["upper"](
            images=imgs, targets=targets, ext_backbone=ext_backbone
        )
        return sum([loss for loss in loss_dict.values()])  # type: ignore
