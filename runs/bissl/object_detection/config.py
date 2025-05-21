from typing import Literal
from source.types import DatasetsDetection

from runs.bissl.config import ArgsBiSSLDefaults
from runs.fine_tune.object_detection.config import ArgsFTDetection


class ArgsBiSSLDetection(ArgsBiSSLDefaults):
    # fmt: off
    #### ADJUSTED DEFAULTS ####
    downstream_task: Literal["object_detection"] = "object_detection"

    d_dataset: DatasetsDetection  # Downstream dataset name
    d_batch_size: int = ArgsFTDetection.batch_size  # Downstream batch size
    d_linear_warmup_epochs: int = 5  # Number of epochs to linearly warmup the downstream model head.

    #### DOWNSTREAM OBJECT DETECTION TASK SPECIFICS ####
    d_img_size_min: int = ArgsFTDetection.img_size_min  # Minimum downstream image size.
    d_img_size_max: int = ArgsFTDetection.img_size_max  # Maximum downstream image size.

    # fmt: on
