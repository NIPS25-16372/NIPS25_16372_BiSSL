from typing import Literal, Optional
from runs.fine_tune.classification.resnet.config import ArgsFTClassificationResNet
from source.types import FloatBetween0and1, OptimizerChoices, UpperLower


class ArgsFTClassificationResNetPostBiSSL(ArgsFTClassificationResNet):
    # fmt: off
    #### ADJUSTED DEFATULS ####
    project_name: Literal["Post_BiSSL_FT"] = "Post_BiSSL_FT"  # Name of the wandb project. Same name is also used for model storage.
    post_pretext_or_bissl: Literal["post-bissl"] = "post-bissl"
    
    img_size: Optional[int] = None  # Image size (img_size x img_size). Defaults to the image size used for pretraining.
    img_crop_min_ratio: Optional[float] = None  # Image crop min ratio. Defaults to the crop min ratio used for pretraining.
    
    dataset: None = None  # (Should not be altered) Datasets. This is meerely a placeholder to be overriden by BiSSL config.
    batch_size: Optional[int] = None  # Batch size
    optimizer: Optional[OptimizerChoices] = None  # Optimizer
    momentum: Optional[FloatBetween0and1] = None # Momentum
    beta1: Optional[FloatBetween0and1] = None  # Beta1 for Adam optimizers
    beta2: Optional[FloatBetween0and1] = None  # Beta2 for Adam optimizers

    #### PostBiSSL SPECIFIC ARGUMENTS ####
    bissl_backbone_origin: UpperLower = "lower" # (Experimental feature, should be left to its default "lower") Origin of the BiSSL backbone. "lower" (/pretext) or "upper" (/downstream).

    # fmt: on
