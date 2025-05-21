from typing import Optional
from source.types import FloatBetween0and1, FloatNonNegative

from runs.bissl.classification.config import ArgsBiSSLClassification


class ArgsBiSSLClassificationSimCLR(ArgsBiSSLClassification):
    # fmt: off
    #### ADJUSTED DEFAULTS ####
    p_lr: FloatNonNegative = 1.0  # SimCLR default learning rate

    #### SimCLR Specific Hyperparameters ####
    p_simclr_temperature: Optional[FloatBetween0and1] = None  # (SimCLR specific) Temperature. Defaults to the temperature used for pretraining.

    # fmt: on
