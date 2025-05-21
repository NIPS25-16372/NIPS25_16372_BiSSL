from typing import Literal
from runs.pretext.config import ArgsPretextDefaults
from source.types import FloatBetween0and1, ConvBackbones


class ArgsPretextSimCLR(ArgsPretextDefaults):
    # fmt: off
    #### ADJUSTED DEFAULTS ####
    # Currently only supports conv backbones
    backbone_arch: ConvBackbones = "resnet50"  # Architecture of the backbone encoder network

    pretext_task: Literal["simclr"] = "simclr"

    #### SimCLR SPECIFIC ARGUMENTS ####
    projector_mlp: str = f"{2**8}-{2**8}-{2**8}"  # Size and number of layers of the MLP expander head
    
    temperature: FloatBetween0and1 = 0.5  # Temperature for the softmax in the contrastive loss

    # fmt: on
