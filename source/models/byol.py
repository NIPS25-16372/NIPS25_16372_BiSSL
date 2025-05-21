import copy
import math
from collections import OrderedDict
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from source.types import ConvBackbones, FloatBetween0and1

from .misc import ConvBackbone, FullGatherLayer, MLPProjector


class EMA:
    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def update_average(self, old, new, beta=None):
        if beta is None:
            beta = self.beta
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new


def update_moving_average(
    ema_updater: EMA,
    ma_model_backbone: torch.nn.Module,
    ma_model_projector: torch.nn.Module,
    current_model_backbone: torch.nn.Module,
    current_model_projector: torch.nn.Module,
    beta: Optional[FloatBetween0and1] = None,
):
    for current_params, ma_params in zip(
        current_model_backbone.parameters(), ma_model_backbone.parameters()
    ):
        old_weight, up_weight = ma_params.data, current_params.data
        ma_params.data = ema_updater.update_average(old_weight, up_weight, beta=beta)

    for current_params, ma_params in zip(
        current_model_projector.parameters(), ma_model_projector.parameters()
    ):
        old_weight, up_weight = ma_params.data, current_params.data
        ma_params.data = ema_updater.update_average(old_weight, up_weight, beta=beta)


def set_requires_grad(model, val):
    for p in model.parameters():
        p.requires_grad = val


class BYOL(nn.Module):
    def __init__(
        self,
        backbone_arch: ConvBackbones,
        proj_mlp: str,
        pred_mlp: Optional[str] = None,
        distributed: bool = True,
        proj_use_bn: bool = True,
        moving_average_decay: FloatBetween0and1 = 0.9995,
        ma_use_scheduler: bool = False,
        ma_scheduler_length: int = 1000,
    ):
        super().__init__()

        self.backbone_arch = backbone_arch
        self.proj_mlp = proj_mlp

        self.backbone, embedding_dim = ConvBackbone(backbone_arch)
        assert embedding_dim is not None

        projector = MLPProjector(proj_mlp, embedding_dim, use_bn=proj_use_bn)
        predictor = MLPProjector(
            pred_mlp or proj_mlp, int(proj_mlp.split("-")[-1]), use_bn=proj_use_bn
        )

        self.head = nn.Sequential(
            OrderedDict([("projector", projector), ("predictor", predictor)])
        )

        self.backbone_target = copy.deepcopy(self.backbone)
        set_requires_grad(self.backbone_target, False)
        self.projector_target = copy.deepcopy(self.head.projector)
        set_requires_grad(self.projector_target, False)

        self.distributed = distributed

        self.target_ema_updater = EMA(moving_average_decay)
        self.ma_decay = moving_average_decay
        self.ma_use_scheduler = ma_use_scheduler
        self.scheduler_step = 1

        # Is equal to the total number of gradient updates, i.e. length of dataloader times number of epochs
        self.ma_scheduler_length = ma_scheduler_length

    def loss_fn(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x = F.normalize(x, dim=-1, p=2)
        y = F.normalize(y, dim=-1, p=2)
        return 2 - 2 * (x * y).sum(dim=-1)

    def update_moving_average(self):
        update_moving_average(
            ema_updater=self.target_ema_updater,
            ma_model_backbone=self.backbone_target,
            ma_model_projector=self.projector_target,
            current_model_backbone=self.backbone,
            current_model_projector=self.head.projector,
            beta=(
                None
                if not self.ma_use_scheduler
                else 1
                - (1 - self.ma_decay)
                * (
                    math.cos(math.pi * self.scheduler_step / self.ma_scheduler_length)
                    + 1
                )
                / 2
            ),
        )
        if self.ma_use_scheduler:
            self.scheduler_step += 1

    def reinit_target_model(self):
        """
        Reinitialize the target model with the current model's weights.
        Necessary for BiSSL when we load a pre-trained model after the model has been initialised.
        """
        self.backbone_target = copy.deepcopy(self.backbone)
        set_requires_grad(self.backbone_target, False)
        self.projector_target = copy.deepcopy(self.head.projector)
        set_requires_grad(self.projector_target, False)

    def forward(
        self, input_pair: Tuple[torch.Tensor, torch.Tensor], update_ema: bool = True
    ) -> torch.Tensor:
        if update_ema:
            self.update_moving_average()

        x, y = input_pair
        x_pred, y_pred = self.head(self.backbone(x)), self.head(self.backbone(y))

        with torch.no_grad():
            x_proj_target, y_proj_target = (
                self.projector_target(self.backbone_target(x)).detach(),
                self.projector_target(self.backbone_target(y)).detach(),
            )

        if self.distributed:
            x_pred, y_pred = torch.cat(FullGatherLayer.apply(x_pred), dim=0), torch.cat(  # type: ignore
                FullGatherLayer.apply(y_pred), dim=0  # type: ignore
            )
            x_proj_target, y_proj_target = torch.cat(
                FullGatherLayer.apply(x_proj_target), dim=0  # type: ignore
            ), torch.cat(
                FullGatherLayer.apply(y_proj_target), dim=0  # type: ignore
            )

        loss_one = self.loss_fn(x_pred, y_proj_target.detach())
        loss_two = self.loss_fn(y_pred, x_proj_target.detach())
        loss = loss_one + loss_two

        return loss.mean()
