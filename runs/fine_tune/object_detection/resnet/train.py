import json
import sys
from typing import Any, Dict, Optional

import torch
import wandb
from ray.air import session

from source.augmentations.downstream import ToTensorMultiInputs
from source.evals import eval_object_detector_ap50
from source.models.downstream import FasterRCNNR50C4BB
from source.trainers import train_object_detector
from source.types import DetectionBackbones, FloatNonNegative

from runs.dataloader import get_dataloader_and_sampler
from runs.fine_tune.misc import (
    init_run,
    wandb_init,
)
from runs.fine_tune.model import generate_model_local_paths
from runs.fine_tune.object_detection.config import ArgsFTDetection
from runs.fine_tune.object_detection.model import load_pretrained_model_and_ddp_wrap
from runs.fine_tune.optimizer import get_optimizer_and_lr_sched
from runs.fine_tune.train import log_data_and_state_dicts
from runs.misc import wandb_watch_models


def train_detection_resnet(
    hpo_config: Dict[str, Any],
    args: ArgsFTDetection,
    backbone_arch: DetectionBackbones,
    pt_config_dict: Dict[str, Any],
    seed: Optional[int] = None,
):

    device = init_run(args, hpo_config, seed=seed)

    wandb_init(args, pretrain_config_dict_to_log=pt_config_dict)

    print("(FT Detection) " + " ".join(sys.argv))
    print(json.dumps(dict(train_type="Console Args", data=" ".join(sys.argv))))
    print("")
    print(f"Device = {device}")

    #### DATA IMPORT ####
    dataloader_train, sampler_train = get_dataloader_and_sampler(
        args=args,
        dataset_name=args.dataset,
        split="train",
        batch_size=args.batch_size,
        transform=ToTensorMultiInputs(),  # type: ignore
        collate_fn=lambda x: tuple(zip(*x)),
    )
    dataloader_val, sampler_val = get_dataloader_and_sampler(
        args=args,
        dataset_name=args.dataset,
        split="val",
        batch_size=args.batch_size,
        transform=ToTensorMultiInputs(),  # type: ignore
        collate_fn=lambda x: tuple(zip(*x)),
    )
    n_classes = len(dataloader_train.dataset.classes)  # type: ignore

    ### MODEL IMPORT ###
    model = FasterRCNNR50C4BB(
        backbone_arch=backbone_arch,
        n_classes=n_classes + 1,
        min_size=args.img_size_min,
        max_size=args.img_size_max,
    )
    load_pretrained_model_and_ddp_wrap(args, model, device)

    ### TRAINING SETUP ###
    assert isinstance(args.wd, FloatNonNegative)
    optimizer, lr_scheduler = get_optimizer_and_lr_sched(
        args=args,
        parameter_groups=[
            dict(
                params=model.parameters(),
                weight_decay=args.wd,
            )
        ],
        len_loader=len(dataloader_train),
    )

    if args.rank == 0:
        assert wandb.run is not None
        wandb_watch_models([model], log_freq=len(dataloader_train))

        if args.save_model:
            model_paths_dict = generate_model_local_paths(
                args=args,
                pretrain_config_dict=pt_config_dict,
                backbone_arch=backbone_arch,
                wandb_run_id=wandb.run.id,
            )

    best_acc = 0

    print("Downstream Fine-Tuning (Object Detection)")
    for epoch in range(int(args.epochs)):

        sampler_train.set_epoch(epoch)
        sampler_val.set_epoch(epoch)

        print("")
        print(f"Epoch {epoch+1} / {args.epochs}\n-------------------------------")

        # Adapt backbone from pretext into downstream model
        loss_tr = train_object_detector(
            model=model,
            dataloader=dataloader_train,
            optimizer=optimizer,
            device=device,
            lr_scheduler=lr_scheduler,
            return_loss_avg_total=True,
        )

        print("")

        acc_te = eval_object_detector_ap50(
            model=model,
            dataloader=dataloader_val,
            device=device,
            label="Val Performance (AP50)",
        )

        best_acc_prev = best_acc
        best_acc = max(best_acc, acc_te)

        session.report({"loss": loss_tr, "accuracy": acc_te})

        if args.rank == 0:
            log_data_and_state_dicts(
                args=args,
                model=model,
                model_paths_dict=model_paths_dict if args.save_model else None,
                tr_stats={"top1": 0, "top5": 0, "loss": loss_tr},  # type: ignore
                te_stats={"top1": acc_te, "top5": 0, "loss": 0},
                lr=optimizer.param_groups[0]["lr"],
                epoch=epoch,
                best_top1_acc=best_acc,
                best_top5_acc=0,
                best_acc_top1_prev=best_acc_prev,
            )

    if args.rank == 0:
        wandb.finish()

    torch.cuda.empty_cache()
    torch.distributed.destroy_process_group()
