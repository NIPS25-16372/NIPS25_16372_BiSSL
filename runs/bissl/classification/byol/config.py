from typing import Optional
from source.types import FloatBetween0and1, BinaryChoices, FloatNonNegative

from runs.bissl.classification.config import ArgsBiSSLClassification


class ArgsBiSSLClassificationBYOL(ArgsBiSSLClassification):
    # fmt: off
    #### ADJUSTED DEFAULTS ####
    p_lr: FloatNonNegative = 1.0  # BYOL default learning rate

    #### BYOL Specific Hyperparameters ####
    p_byol_ma_decay: Optional[FloatBetween0and1] = None  # (BYOL specific) Moving average decay. Defaults to the moving average decay used for pretraining.
    p_byol_ma_use_scheduler: Optional[BinaryChoices] = None  # (BYOL specific) Whether to use a moving average scheduler. Defaults to the moving average scheduler used for pretraining.

    # fmt: on
