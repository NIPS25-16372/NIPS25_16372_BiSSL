from typing import Optional, Literal

from runs.fine_tune.config import ArgsFineTuningDefaults
from source.types import DatasetsDetection, FloatNonNegative


class ArgsFTDetection(ArgsFineTuningDefaults):
    # fmt: off
    #### ADJUSTED DEFAULTS ####
    # General Training Args
    downstream_task: Literal["object_detection"] = "object_detection"  # Type of downstream task.
    dataset: DatasetsDetection  # Datasets
    batch_size: int = 16  # Batch size
    epochs: int = 50  # Number of training epochs
    
    # HPO Args
    hpo_lr_min: Optional[FloatNonNegative] = 1e-4  # Min learning rate for HPO
    hpo_lr_max: Optional[FloatNonNegative] = 1e-1  # Max learning rate for HPO
    hpo_wd_min: Optional[FloatNonNegative] = 1e-6  # Min weight decay for HPO
    hpo_wd_max: Optional[FloatNonNegative] = 1e-2  # Max weight decay for HPO
    num_runs: int = 50  # Number of HPO runs to perform

    #### DETECTION SPECIFIC ARGUMENTS ####
    img_size_min: int = 196  # Minimum downstream image edge size.
    img_size_max: int = 320  # Maximum downstream image edge size.

    # fmt: on
