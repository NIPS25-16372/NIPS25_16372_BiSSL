from typing import Tuple, Dict
import os
import json
import torch
from runs.config_general import CONFIG_KEYS_IGNORE_WHEN_SAVING
from runs.pretext.config import ArgsPretextDefaults


### MODEL IMPORT ###
def model_to_ddp(
    model: torch.nn.Module,
    device: torch.device,
) -> torch.nn.Module:
    print(
        f"Backbone: {sum(p.numel() for p in model.backbone.parameters() if p.requires_grad)/1_000_000} M Parameters"
    )
    print(
        f"Pretext Head: {sum(p.numel() for p in model.head.parameters() if p.requires_grad)/1_000_000} M Parameters"
    )

    model.to(device)
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = torch.nn.parallel.DistributedDataParallel(model)

    model.backbone = model.module.backbone
    model.head = model.module.head

    return model


def generate_model_local_paths(
    args: ArgsPretextDefaults,
    wandb_run_id: str,
) -> Dict[str, str]:
    model_root = args.model_root or args.root + "/models/"

    assert os.path.isdir(model_root), "Model root directory does not exist."

    mod_local_path_prefix = (
        model_root
        + args.project_name
        + "_"
        + args.pretext_task
        + "_arch-"
        + args.backbone_arch
    )

    mod_local_path_bb = mod_local_path_prefix + "_backbone_id-" + wandb_run_id + ".pth"
    mod_local_path_head = mod_local_path_prefix + "_head_id-" + wandb_run_id + ".pth"

    model_paths_dict = {
        "bb": mod_local_path_bb,
        "head": mod_local_path_head,
    }

    # Storing the model config into a similarly named json file
    model_config_path = mod_local_path_prefix + "_config_id-" + wandb_run_id + ".json"
    args_dict = args.as_dict()

    # Remove keys that are not relevant for the model
    args_dict = {
        k: v for k, v in args_dict.items() if k not in CONFIG_KEYS_IGNORE_WHEN_SAVING
    }
    with open(model_config_path, "w") as fp:
        json.dump(args_dict, fp, sort_keys=True, indent=4)

    return model_paths_dict


def input_processor_tupled(
    x: Tuple[Tuple[torch.Tensor, torch.Tensor], torch.Tensor],
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    (x1, x2), _ = x
    return x1.to(device, non_blocking=True), x2.to(device, non_blocking=True)


def input_processor_single(
    x: Tuple[torch.Tensor, torch.Tensor], device: torch.device
) -> torch.Tensor:
    return x[0].to(device, non_blocking=True)
