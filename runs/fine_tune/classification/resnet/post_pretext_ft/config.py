from typing import Literal
from runs.fine_tune.classification.resnet.config import ArgsFTClassificationResNet


class ArgsFTClassificationResNetPostPretext(ArgsFTClassificationResNet):
    # fmt: off
    #### ADJUSTED DEFATULS ####
    project_name: Literal["Post_Pretext_FT"] = "Post_Pretext_FT"  # Name of the wandb project. Same name is also used for model storage.
    post_pretext_or_bissl: Literal["post-pretext"] = "post-pretext"

    #### PostPretext SPECIFIC ARGUMENTS ####

    # fmt: on
