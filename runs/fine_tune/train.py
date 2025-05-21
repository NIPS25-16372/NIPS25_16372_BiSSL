from typing import Dict, Literal, Optional
import wandb
import torch


def log_stats_to_wandb(
    tr_stats: Dict[Literal["top1", "top5", "loss"], float],
    te_stats: Dict[Literal["top1", "top5", "loss"], float],
    lr: float,
    epoch: int,
    best_top1_acc: float,
    best_top5_acc: float,
):
    wandb.log(
        {
            "train_ft/top1_acc": tr_stats["top1"],
            "train_ft/top5_acc": tr_stats["top5"],
            "train_ft/avgloss": tr_stats["loss"],
            "test_ft/top1_acc": te_stats["top1"],
            "test_ft/top5_acc": te_stats["top5"],
            "test_ft/avgloss": te_stats["loss"],
            "test_ft/best_top1_acc": best_top1_acc,
            "test_ft/best_top5_acc": best_top5_acc,
            "test_ft/lr": lr,
            "test_ft/epoch": epoch + 1,
        }
    )


def store_state_dict(
    model: torch.nn.Module,
    model_paths_dict: Dict[str, str],
):
    model.eval()

    sd_bb = model.backbone.state_dict()
    sd_h = model.head.state_dict()

    torch.save(sd_bb, model_paths_dict["bb"])
    torch.save(sd_h, model_paths_dict["h"])


def log_data_and_state_dicts(
    args,
    model: torch.nn.Module,
    model_paths_dict: Optional[Dict[str, str]],
    tr_stats: Dict[Literal["top1", "top5", "loss"], float],
    te_stats: Dict[Literal["top1", "top5", "loss"], float],
    lr: float,
    epoch: int,
    best_top1_acc: float,
    best_top5_acc: float,
    best_acc_top1_prev: float,
):
    assert tr_stats.keys() == te_stats.keys() == {"top1", "top5", "loss"}
    log_stats_to_wandb(
        tr_stats=tr_stats,
        te_stats=te_stats,
        lr=lr,
        epoch=epoch,
        best_top1_acc=best_top1_acc,
        best_top5_acc=best_top5_acc,
    )

    if bool(args.save_model) and (best_top1_acc - best_acc_top1_prev) > 0:
        assert model_paths_dict is not None
        store_state_dict(model, model_paths_dict)
        print("Model State Dicts Saved.")
