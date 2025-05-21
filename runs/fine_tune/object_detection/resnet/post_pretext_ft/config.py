from typing import Literal

from runs.fine_tune.object_detection.config import ArgsFTDetection


class ArgsFTDetectionPostPretext(ArgsFTDetection):
    # fmt: off
    #### ADJUSTED DEFAULTS ####
    project_name: Literal["Post_Pretext_FT"] = "Post_Pretext_FT"  # Name of the wandb project. Same name is also used for model storage.
    post_pretext_or_bissl: Literal["post-pretext"] = "post-pretext"

    #### Post-Pretext SPECIFIC ARGUMENTS ####

    # fmt: on
