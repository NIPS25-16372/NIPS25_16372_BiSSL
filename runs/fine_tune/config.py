from typing import Optional, Literal

from runs.config_general import ArgsGeneralDefaults
from source.types import (
    BinaryChoices,
    DownstreamDatasets,
    DownstreamTaskTypes,
    FloatBetween0and1,
    FloatNonNegative,
    OptimizerChoices,
)


class ArgsFineTuningDefaults(ArgsGeneralDefaults):
    # fmt: off
    pretrained_model_backbone: str  # Name of .pth file with pretrained model backbone weights, e.g. "BiSSL_simclr_classification_pets_arch-resnet50_backbone_id-tii8s7xv.pth"
    pretrained_model_config: str # Name of .json file with pretrained model config, e.g. "BiSSL_simclr_classification_pets_arch-resnet50_config_id-tii8s7xv.json"
    
    post_pretext_or_bissl: Literal["post-pretext", "post-bissl"]  # Specify if the run is post-pretext or post-bissl. Should be overridden by the specific run config.
    # Dataset Args
    downstream_task: DownstreamTaskTypes  # Type of downstream task
    dataset: DownstreamDatasets  # Dataset
    # use_randaug: BinaryChoices = 0 # Use RandAugment

    # General Training Args
    lr: Optional[FloatNonNegative] = None  # Learning rate. Is overridden by HPO if use_hpo is set to 1.
    wd: Optional[FloatNonNegative] = None  # Weight decay. Is overridden by HPO if use_hpo is set to 1.

    batch_size: int  # Batch size
    epochs: int  # Number of training epochs
    optimizer: OptimizerChoices = "sgd"  # Optimizer
    momentum: FloatBetween0and1 = 0.9  # Momentum
    beta1: FloatBetween0and1 = 0.9  # Beta1 for Adam optimizers
    beta2: FloatBetween0and1 = 0.999  # Beta2 for Adam optimizers

    # Model args
    save_model: BinaryChoices = 0  # Save model

    # HPO Args
    use_hpo: BinaryChoices = 1  # Use hyperparameter optimization. If
    hpo_lr_min: Optional[FloatNonNegative] = 0.0001  # Min learning rate for HPO
    hpo_lr_max: Optional[FloatNonNegative] = 1.0  # Max learning rate for HPO
    hpo_wd_min: Optional[FloatNonNegative] = 0.00001  # Min weight decay for HPO
    hpo_wd_max: Optional[FloatNonNegative] = 0.01  # Max weight decay for HPO

    num_runs: int = 100  # Number of HPO runs to perform

    # fmt: on
