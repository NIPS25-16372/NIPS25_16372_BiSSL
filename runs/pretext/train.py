from typing import Dict, List
import json
import wandb
import torch


def log_data_and_state_dicts(
    log_dicts: List[Dict[str, float]],
    model: torch.nn.Module,
    model_paths_dict: Dict[str, str],
    save_model: bool = True,
):

    if log_dicts is not None:
        for dic in log_dicts:
            wandb.log(dic)

    if save_model:
        model.eval()

        torch.save(
            model.backbone.state_dict(),
            model_paths_dict["bb"],
        )
        torch.save(
            model.head.state_dict(),
            model_paths_dict["head"],
        )
