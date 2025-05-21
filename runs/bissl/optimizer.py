from typing import List, Dict, Iterator, Tuple
import torch
from source.optimizers import get_optimizer
from source.schedulers import CosineLRSchedulerWithWarmup

from runs.bissl.config import ArgsBiSSLDefaults
from runs.pretext.config import ArgsPretextDefaults


def get_lower_pretext_optimizer_and_lr_sched(
    args: ArgsBiSSLDefaults,
    pretrain_config: ArgsPretextDefaults,
    parameter_groups: (
        List[Dict[str, Iterator[torch.nn.Parameter] | float]]
        | List[Dict[str, List[torch.nn.Parameter] | float]]
    ),
) -> Tuple[torch.optim.Optimizer, CosineLRSchedulerWithWarmup]:

    assert all(
        ("params" in pg.keys() and "weight_decay" in pg.keys())
        for pg in parameter_groups
    )

    optimizer = get_optimizer(
        args.p_optimizer or pretrain_config.optimizer,
        parameter_groups,
        lr=args.p_lr or pretrain_config.lr,
        momentum=args.p_momentum or pretrain_config.momentum,
        betas=(
            args.p_beta1 or pretrain_config.beta1,
            args.p_beta2 or pretrain_config.beta2,
        ),
    )

    lr_scheduler = CosineLRSchedulerWithWarmup(
        optimizer=optimizer,
        n_epochs=0,  # args.epochs * args.p_iter_epoch + add_epochs,
        len_loader=0,
        warmup_epochs=0,
    )
    # As the warmup and remainder of training have different number of steps,
    # we need to set the total_steps and warmup_steps manually
    lr_scheduler.total_steps = args.lower_num_iter * args.epochs
    lr_scheduler.warmup_steps = 10 * args.lower_num_iter

    return optimizer, lr_scheduler


def get_upper_downstream_optimizer_and_lr_sched(
    args: ArgsBiSSLDefaults,
    parameter_groups: List[
        Dict[str, Iterator[torch.nn.Parameter] | List[torch.nn.Parameter] | float]
    ],
    d_lr_sched_warmup_epochs: int = 0,
):

    assert all(
        ("params" in pg.keys() and "weight_decay" in pg.keys())
        for pg in parameter_groups
    )

    optimizer = get_optimizer(
        args.d_optimizer,
        parameter_groups,
        lr=args.d_lr,
        momentum=args.d_momentum,
        betas=(args.d_beta1, args.d_beta2),
    )

    lr_scheduler = CosineLRSchedulerWithWarmup(
        optimizer=optimizer,
        n_epochs=args.epochs,
        len_loader=args.upper_num_iter,
        warmup_epochs=d_lr_sched_warmup_epochs,
    )

    return optimizer, lr_scheduler
