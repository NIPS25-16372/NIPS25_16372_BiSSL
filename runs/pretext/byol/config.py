from typing import Literal, Optional
from runs.pretext.config import ArgsPretextDefaults
from source.types import FloatBetween0and1, BinaryChoices, ConvBackbones


class ArgsPretextBYOL(ArgsPretextDefaults):
    # fmt: off
    #### ADJUSTED DEFAULTS ####
    # Currently only supports conv backbones
    backbone_arch: ConvBackbones = "resnet50"  # Architecture of the backbone encoder network

    pretext_task: Literal["byol"] = "byol"

    #### BYOL SPECIFIC ARGUMENTS ####
    projector_mlp: str = f"{2**8}-{2**8}-{2**8}"  # Size and number of layers of the MLP projection head. Is formatted as a string with the number of neurons in each layer separated by a hyphen. E.g. "64-256-512"
    predictor_mlp: Optional[str] = None  # Size and number of layers of the MLP predictor head, formatted as projector_mlp. If None (the default), the predictor head is set to be architecturally identical to the projector head.
    
    ma_decay: FloatBetween0and1 = 0.9995  # Decay rate for the moving average of the target encoder
    ma_use_scheduler: BinaryChoices = 0  # Use a scheduler for the moving average decay rate

    # fmt: on
