from typing import Dict, Literal, Tuple

import torch
from torch.utils.data import DataLoader

from source.evals import eval_classifier_mAP, eval_classifier_top1_and_top5

from runs.fine_tune.classification.config import ArgsFTClassification


def ft_eval_classifier_single(
    args: ArgsFTClassification,
    model: torch.nn.Module,
    dataloader: DataLoader,
    loss_fn: torch.nn.Module,
    device: torch.device,
    label: str = "Train Performance",
    non_blocking: bool = True,
) -> Dict[Literal["top1", "top5", "loss"], float]:
    if args.dataset == "voc07":
        top1 = eval_classifier_mAP(
            model=model,
            dataloader=dataloader,
            device=device,
            label=label + " (mAP)",
            non_blocking=non_blocking,
        )

        top5, loss = 0.0, 0.0
    else:
        top1, top5, loss = eval_classifier_top1_and_top5(
            model=model,
            dataloader=dataloader,
            loss_fn=loss_fn,
            device=device,
            label=label + " (Top1 and Top5)",
            non_blocking=non_blocking,
        )

    stats: Dict[Literal["top1", "top5", "loss"], float] = {
        "top1": top1,
        "top5": top5,
        "loss": loss,
    }
    return stats


def ft_eval_classifier(
    args: ArgsFTClassification,
    model: torch.nn.Module,
    dataloader_train: DataLoader,
    dataloader_val: DataLoader,
    loss_fn: torch.nn.Module,
    device: torch.device,
    label: str = "Train Performance",
    non_blocking: bool = True,
) -> Tuple[
    Dict[Literal["top1", "top5", "loss"], float],
    Dict[Literal["top1", "top5", "loss"], float],
]:
    train_stats: Dict[Literal["top1", "top5", "loss"], float] = (
        ft_eval_classifier_single(
            args=args,
            model=model,
            dataloader=dataloader_train,
            loss_fn=loss_fn,
            device=device,
            label=label,
            non_blocking=non_blocking,
        )
    )
    test_stats: Dict[Literal["top1", "top5", "loss"], float] = (
        ft_eval_classifier_single(
            args=args,
            model=model,
            dataloader=dataloader_val,
            loss_fn=loss_fn,
            device=device,
            label=label,
            non_blocking=non_blocking,
        )
    )

    return train_stats, test_stats
