from typing import List, Dict, Iterator
import torch
from source.optimizers import get_optimizer
from source.trainers import CosineLRSchedulerWithWarmup
from runs.pretext.config import ArgsPretextDefaults


def get_pretext_optimizer_and_lr_sched(
    args: ArgsPretextDefaults,
    parameter_groups: (
        List[Dict[str, Iterator[torch.nn.Parameter] | float]]
        | List[Dict[str, List[torch.nn.Parameter] | float]]
    ),
    dataloader_len: int,
    lr_sched_warmup_epochs: int = 10,
):

    assert all(
        ("params" in pg.keys() and "weight_decay" in pg.keys())
        for pg in parameter_groups
    )

    optimizer = get_optimizer(
        args.optimizer,
        parameter_groups,
        lr=args.lr,
        momentum=args.momentum,
        betas=(args.beta1, args.beta2),
    )

    lr_scheduler = CosineLRSchedulerWithWarmup(
        optimizer=optimizer,
        n_epochs=args.epochs,
        len_loader=dataloader_len,
        warmup_epochs=lr_sched_warmup_epochs,
    )

    return optimizer, lr_scheduler
