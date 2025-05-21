from typing import Optional
from source.types import FloatBetween0and1, BinaryChoices, FloatNonNegative

from runs.bissl.object_detection.config import ArgsBiSSLDetection


class ArgsBiSSLDetectionBYOL(ArgsBiSSLDetection):
    # fmt: off
    #### ADJUSTED DEFAULTS ####
    d_dataset = "voc07+12detection"
    d_lr: FloatNonNegative = 0.025
    d_wd: FloatNonNegative = 0.0001
    p_lr: FloatNonNegative = 1.0  # BYOL default learning rate

    #### BYOL Specific Hyperparameters ####
    p_byol_ma_decay: Optional[FloatBetween0and1] = None  # (BYOL specific) Moving average decay. Defaults to the moving average decay used for pretraining.
    p_byol_ma_use_scheduler: Optional[BinaryChoices] = None  # (BYOL specific) Whether to use a moving average scheduler. Defaults to the moving average scheduler used for pretraining.

    # fmt: on
