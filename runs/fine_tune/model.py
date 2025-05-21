import json
from typing import Dict, Any
import os
from runs.config_general import CONFIG_KEYS_IGNORE_WHEN_SAVING
from runs.fine_tune.config import ArgsFineTuningDefaults
from source.types import BackboneArchs


def generate_model_local_paths(
    args: ArgsFineTuningDefaults,
    pretrain_config_dict: Dict[str, Any],
    backbone_arch: BackboneArchs,
    wandb_run_id: str,
):

    model_root = args.model_root or args.root + "/models"

    assert os.path.isdir(model_root), "Model root directory does not exist."

    mod_local_path_prefix = (
        model_root
        + "/"
        + args.project_name
        + "_"
        + args.downstream_task
        + "_"
        + args.dataset
        + "_arch-"
        + backbone_arch
    )

    mod_local_path_bb = mod_local_path_prefix + "_backbone_id-" + wandb_run_id + ".pth"
    mod_local_path_h = mod_local_path_prefix + "_head_id-" + wandb_run_id + ".pth"

    model_paths_dict = {"bb": mod_local_path_bb, "h": mod_local_path_h}

    # Storing the model config into a similarly named json file
    model_config_path = mod_local_path_prefix + "_config_id-" + wandb_run_id + ".json"
    args_dict = args.as_dict()

    # Remove keys that are not relevant for the model
    args_dict = {
        k: v for k, v in args_dict.items() if k not in CONFIG_KEYS_IGNORE_WHEN_SAVING
    }
    # Also includes the pretrain config
    pt_config = {
        pt_stage: {
            k: v for k, v in pt_dict.items() if k not in CONFIG_KEYS_IGNORE_WHEN_SAVING
        }
        for pt_stage, pt_dict in pretrain_config_dict.items()
    }
    args_dict = args_dict | pt_config
    # Stores config
    with open(model_config_path, "w") as fp:
        json.dump(args_dict, fp, sort_keys=True, indent=4)

    return model_paths_dict
