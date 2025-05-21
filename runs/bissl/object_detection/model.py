from collections import OrderedDict
import copy
from typing import Callable, Dict, List, Tuple
import torch

from source.models.simclr import SimCLR
from source.models.byol import BYOL


def get_upper_input_processor(
    device: torch.device,
) -> Callable:
    def return_fn(
        upper_input: Tuple[List[torch.Tensor], List[Dict[str, torch.Tensor]]],
    ) -> Tuple[List[torch.Tensor], List[Dict[str, torch.Tensor]]]:
        imgs, labels = upper_input
        return [img.to(device, non_blocking=True) for img in imgs], [
            {k: v.to(device, non_blocking=True) for k, v in t.items()} for t in labels
        ]

    return return_fn


def wrap_simclr_r50c4(
    model: SimCLR,
) -> SimCLR:

    assert model.backbone_arch == "resnet50"

    # The head now includes the final layer of the backbone
    projector = copy.deepcopy(model.head.projector)
    model.head = torch.nn.Sequential(
        OrderedDict(
            [
                ("layer4", copy.deepcopy(model.backbone.layer4)),
                ("global_pool", copy.deepcopy(model.backbone.global_pool)),
                ("fc", copy.deepcopy(model.backbone.fc)),
                ("projector", projector),
            ]
        )
    )

    # The backbone terminates at layer4, converting it to a R50C4 backbone.
    model.backbone.layer4 = torch.nn.Identity()
    del model.backbone.global_pool
    del model.backbone.fc
    model.backbone.forward = model.backbone.forward_features

    return model


def wrap_byol_r50c4(
    model: BYOL,
) -> BYOL:

    assert model.backbone_arch == "resnet50"

    # The head now includes the final layer of the backbone.
    # We update the projector to include this.
    projector = copy.deepcopy(model.head.projector)
    predictor = copy.deepcopy(model.head.predictor)

    model.head = torch.nn.Sequential(
        OrderedDict(
            [
                ("layer4", copy.deepcopy(model.backbone.layer4)),
                ("global_pool", copy.deepcopy(model.backbone.global_pool)),
                ("fc", copy.deepcopy(model.backbone.fc)),
                ("projector", projector),
                ("predictor", predictor),
            ]
        )
    )

    # The backbone terminates at layer4, converting it to a R50C4 backbone.
    model.backbone.layer4 = torch.nn.Identity()
    del model.backbone.global_pool
    del model.backbone.fc
    model.backbone.forward = model.backbone.forward_features

    # Updates the target backbone and projector to match the updated model.
    model.reinit_target_model()

    return model


def pretext_bb_r50_to_r50c4(model: torch.nn.Module):
    if isinstance(model, SimCLR):
        return wrap_simclr_r50c4(model)
    elif isinstance(model, BYOL):
        return wrap_byol_r50c4(model)
    else:
        raise ValueError(f"Unsupported model type: {type(model)}")
