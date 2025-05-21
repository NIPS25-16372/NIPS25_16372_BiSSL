from typing import Optional
from source.types import FloatBetween0and1, FloatNonNegative

from runs.bissl.object_detection.config import ArgsBiSSLDetection


class ArgsBiSSLDetectionSimCLR(ArgsBiSSLDetection):
    # fmt: off
    #### ADJUSTED DEFAULTS ####
    d_dataset = "voc07+12detection"
    d_lr : FloatNonNegative = 0.015
    d_wd: float = 0.0
    p_lr: FloatNonNegative = 1.0  # BYOL default learning rate

    #### SimCLR Specific Hyperparameters ####
    p_simclr_temperature: Optional[FloatBetween0and1] = None  # (SimCLR specific) Temperature. Defaults to the temperature used for pretraining.

    # fmt: on
