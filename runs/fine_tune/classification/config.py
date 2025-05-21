from typing import Optional, Literal

from runs.fine_tune.config import ArgsFineTuningDefaults
from runs.pretext.config import ArgsPretextDefaults
from source.types import DatasetsClassification, FloatNonNegative


class ArgsFTClassification(ArgsFineTuningDefaults):
    # fmt: off
    #### ADJUSTED DEFATULS ####
    downstream_task: Literal["classification"] = "classification"
    dataset: DatasetsClassification  # Dataset
    batch_size: int = 256  # Batch size
    epochs: int = 400  # Number of training epochs
    
    img_size: int = ArgsPretextDefaults.img_size  # Image size (img_size x img_size). Defaults to the image size used for pretraining.
    img_crop_min_ratio: float = ArgsPretextDefaults.img_crop_min_ratio  # Image crop min ratio. Defaults to the crop min ratio used for pretraining.
    
    # HPO Args
    hpo_lr_min: Optional[FloatNonNegative] = 1e-4  # Min learning rate for HPO
    hpo_lr_max: Optional[FloatNonNegative] = 1e0  # Max learning rate for HPO
    hpo_wd_min: Optional[FloatNonNegative] = 1e-5  # Min weight decay for HPO
    hpo_wd_max: Optional[FloatNonNegative] = 1e-2  # Max weight decay for HPO
    num_runs: int = 200  # Number of HPO runs to perform

    #### CLASSIFICAITON SPECIFIC ARGUMENTS ####
    img_size: int = ArgsPretextDefaults.img_size  # Image size (img_size x img_size). Defaults to the image size used for pretraining.
    img_crop_min_ratio: float = ArgsPretextDefaults.img_crop_min_ratio  # Image crop min ratio. Defaults to the crop min ratio used for pretraining.
    # fmt: on
