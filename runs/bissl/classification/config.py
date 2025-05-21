from typing import Optional, Literal
from source.types import (
    DatasetsClassification,
)

from runs.bissl.config import ArgsBiSSLDefaults
from runs.fine_tune.classification.config import ArgsFTClassification


class ArgsBiSSLClassification(ArgsBiSSLDefaults):
    # fmt: off
    #### ADJUSTED DEFAULTS ####
    downstream_task: Literal["classification"] = "classification"

    d_dataset: DatasetsClassification  # Downstream dataset name
    d_batch_size: int = ArgsFTClassification.batch_size  # Downstream batch size
    # d_use_randaug: BinaryChoices = 0  # Whether to use RandAugment for downstream training

    #### DOWNSTREAM CLASSIFICATION TASK SPECIFICS ####
    d_img_size: Optional[int] = None  # Downstream image size (d_img_size x d_img_size). Defaults to the image size used for pretraining.
    d_img_crop_min_ratio: Optional[float] = None  # Downstream image crop min ratio. Defaults to the crop min ratio used for pretraining.

    # fmt: on
