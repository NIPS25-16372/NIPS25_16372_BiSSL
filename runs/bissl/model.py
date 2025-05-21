import os
import json
from typing import Callable, Tuple
import torch

from runs.bissl.config import ArgsBiSSLDefaults
from runs.config_general import CONFIG_KEYS_IGNORE_WHEN_SAVING
from runs.pretext.config import ArgsPretextDefaults


def wrap_model_for_ddp(
    model: torch.nn.Module,
    device: torch.device,
    find_unused_parameters: bool = False,
) -> torch.nn.Module:
    model = torch.nn.parallel.DistributedDataParallel(
        model.to(device), find_unused_parameters=find_unused_parameters
    )

    model.backbone = model.module.backbone
    model.head = model.module.head

    torch.distributed.barrier()

    return model


def generate_model_local_paths(
    args: ArgsBiSSLDefaults,
    pretrain_config: ArgsPretextDefaults,
    wandb_run_id: str,
):
    model_root = args.model_root or args.root + "/models"

    assert os.path.isdir(model_root), "Model root directory does not exist."

    mod_path_local_prefix = (
        model_root
        + "/"
        + args.project_name
        + "_"
        + pretrain_config.pretext_task
        + "_"
        + args.downstream_task
        + "_"
        + args.d_dataset
        + "_arch-"
        + pretrain_config.backbone_arch
    )

    mod_path_p_bb = mod_path_local_prefix + f"_lower_backbone_id-{wandb_run_id}.pth"

    model_paths_dict = {
        "p_bb": mod_path_p_bb,
    }

    # Storing the model config into a similarly named json file
    model_config_path = mod_path_local_prefix + f"_config_id-{wandb_run_id}.json"
    args_dict = args.as_dict()

    # Remove keys that are not relevant for the model
    args_dict = {
        k: v for k, v in args_dict.items() if k not in CONFIG_KEYS_IGNORE_WHEN_SAVING
    }

    # Also includes the pretrain config
    pretrain_config_dict = pretrain_config.as_dict()
    pretrain_config_dict = {
        k: v
        for k, v in pretrain_config_dict.items()
        if k not in CONFIG_KEYS_IGNORE_WHEN_SAVING
    }

    # Includes the pretrain config as a sub-dictionary
    args_dict["PT"] = pretrain_config_dict

    # Stores config
    with open(model_config_path, "w") as fp:
        json.dump(args_dict, fp, sort_keys=True, indent=4)

    return model_paths_dict


def load_pretext_pretrained_model(
    args: ArgsBiSSLDefaults,
    pretrain_config: ArgsPretextDefaults,
    model_p: torch.nn.Module,
) -> torch.nn.Module:
    assert not next(model_p.parameters()).is_cuda
    if args.rank == 0:

        model_root = args.model_root or args.root + "/models"

        model_p.backbone.load_state_dict(
            torch.load(model_root + "/" + args.pretrained_model_backbone)
        )
        model_p.head.load_state_dict(
            torch.load(model_root + "/" + args.pretrained_model_head)
        )

        if pretrain_config.pretext_task == "byol":
            # Re-initialises target model s.t. its parameters are equal to the online model.
            model_p.reinit_target_model()

    return model_p


def get_lower_input_processor_tupled(
    device: torch.device,
) -> Callable:
    """
    Returns a function takes a tuple with input data and labels respectively, and returns the processed input data.
    Compatible with pretext tasks that operate on two augmented images such as SimCLR or BYOL.
    """

    def return_fn(
        x: Tuple[Tuple[torch.Tensor, torch.Tensor], torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        (x1, x2), _ = x
        return x1.to(device, non_blocking=True), x2.to(device, non_blocking=True)

    return return_fn


def get_lower_input_processor_single(
    device: torch.device,
) -> Callable:
    """
    Returns a function that processes a tuple of input data and labels respectively, and returns the processed input data.
    Compatible with pretext tasks that operate on a single image, such as MAEs.
    """

    def return_fn(x: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        img, _ = x
        return img.to(device, non_blocking=True)

    return return_fn
