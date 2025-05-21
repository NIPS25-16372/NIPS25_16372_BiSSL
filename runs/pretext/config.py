from typing import Literal
from runs.config_general import ArgsGeneralDefaults
from source.types import (
    BackboneArchs,
    PretextTaskTypes,
    PretrainDatasets,
    OptimizerChoices,
    FloatNonNegative,
    FloatBetween0and1,
    BinaryChoices,
)


class ArgsPretextDefaults(ArgsGeneralDefaults):
    # fmt: off
    
    project_name: Literal["Pretext"] = "Pretext"  # Name of the overall project. Mainly used for logging purposes to wandb.

    # This should be overwritten to only be forced to be only one possible value, determined by the pretext task.
    pretext_task: PretextTaskTypes  # Type of pretext task.

    img_size: int = 96  # Size of the input images (img_size x img_size)
    img_crop_min_ratio: float = 0.5  # Minimum crop ratio for the random crop

    backbone_arch: BackboneArchs = "resnet50"  # Architecture of the backbone encoder network

    epochs: int = 500  # Number of epochs
    dataset: PretrainDatasets = "imagenet"  # Dataset
    batch_size: int = 2**10 # Effective batch size (per worker batch size is [batch-size] / world-size)

    optimizer: OptimizerChoices = "lars"  # Optimizer
    lr: FloatNonNegative = 4.8  # Base learning  rate
    wd: FloatNonNegative = 1e-6  # Weight decay
    momentum: FloatBetween0and1 = 0.9  # Momentum
    beta1: FloatBetween0and1 = 0.9  # Beta1 for Adam
    beta2: FloatBetween0and1 = 0.999  # Beta2 for Adam

    save_model: BinaryChoices = 1  # Save model to storage.

    # fmt: on
