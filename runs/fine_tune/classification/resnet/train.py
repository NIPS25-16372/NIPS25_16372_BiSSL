from typing import Any, Dict, Optional
import json
import sys

import torch
import torch.distributed
import wandb
from ray.air import session

from source.augmentations.downstream import (
    DownstreamClassificationTrainTransform,
    DownstreamClassificaitonTestTransform,
)
from source.models.downstream import DSClassifier
from source.trainers import train_classifier
from source.types import ConvBackbones

from runs.dataloader import get_dataloader_and_sampler
from runs.misc import wandb_watch_models

from runs.fine_tune.misc import init_run, wandb_init
from runs.fine_tune.model import generate_model_local_paths
from runs.fine_tune.optimizer import get_optimizer_and_lr_sched
from runs.fine_tune.train import log_data_and_state_dicts

from runs.fine_tune.classification.resnet.config import ArgsFTClassificationResNet
from runs.fine_tune.classification.train import ft_eval_classifier


def train_classification_resnet(
    hpo_config: Dict[str, Any],
    args: ArgsFTClassificationResNet,
    backbone_arch: ConvBackbones,
    pt_config_dict: Dict[str, Any],
    seed: Optional[int] = None,
):

    device = init_run(args, hpo_config, seed=seed)

    wandb_init(args, pretrain_config_dict_to_log=pt_config_dict)

    print("(FT Classification) " + " ".join(sys.argv))
    print(json.dumps(dict(train_type="Console Args", data=" ".join(sys.argv))))
    print("")
    print(f"Device = {device}")

    #### DATA IMPORT ####
    dataloader_train, sampler_train = get_dataloader_and_sampler(
        args=args,
        dataset_name=args.dataset,
        split="train",
        batch_size=args.batch_size,
        transform=DownstreamClassificationTrainTransform(
            img_size=args.img_size,
            min_ratio=args.img_crop_min_ratio,
        ),
    )
    dataloader_val, sampler_val = get_dataloader_and_sampler(
        args=args,
        dataset_name=args.dataset,
        split="val",
        batch_size=args.batch_size,
        transform=DownstreamClassificaitonTestTransform(
            img_size=args.img_size,
        ),
    )
    n_classes = len(dataloader_train.dataset.classes)  # type: ignore

    ### MODEL IMPORT ###
    pretrained_bb_path = (
        (args.model_root or args.root + "/models")
        + "/"
        + args.pretrained_model_backbone
    )

    model = DSClassifier(
        backbone_arch=backbone_arch,
        n_classes=n_classes,
    )
    model.backbone.load_state_dict(torch.load(pretrained_bb_path, map_location=device))
    model.to(device)
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = torch.nn.parallel.DistributedDataParallel(model)

    torch.distributed.barrier()
    model.backbone = model.module.backbone
    model.head = model.module.head

    ### TRAINING SETUP ###
    loss_fn = torch.nn.CrossEntropyLoss()

    optimizer, lr_scheduler = get_optimizer_and_lr_sched(
        args=args,
        parameter_groups=[
            dict(
                params=model.parameters(),
                weight_decay=args.wd,  # type: ignore
            )
        ],
        len_loader=len(dataloader_train),
    )

    if args.rank == 0:
        assert wandb.run is not None
        wandb_watch_models([model], log_freq=len(dataloader_train))

        if bool(args.save_model):
            model_paths_dict = generate_model_local_paths(
                args=args,
                pretrain_config_dict=pt_config_dict,
                backbone_arch=backbone_arch,
                wandb_run_id=wandb.run.id,
            )

    best_acc_top1 = 0.0
    best_acc_top5 = 0.0

    print("'Downstream Fine-Tuning (Classification)")
    for epoch in range(int(args.epochs)):

        sampler_train.set_epoch(epoch)
        sampler_val.set_epoch(epoch)

        print("")
        print(f"Epoch {epoch+1} / {args.epochs}\n-------------------------------")

        # Adapt backbone from pretext into downstream model
        train_classifier(
            model=model,
            loss_fn=loss_fn,
            dataloader=dataloader_train,
            optimizer=optimizer,
            device=device,
            lr_scheduler=lr_scheduler,
        )

        print("")

        train_stats, test_stats = ft_eval_classifier(
            args=args,
            model=model,
            dataloader_train=dataloader_train,
            dataloader_val=dataloader_val,
            loss_fn=loss_fn,
            device=device,
        )

        best_acc_top1_prev = best_acc_top1
        best_acc_top1 = max(best_acc_top1, test_stats["top1"])
        best_acc_top5 = max(best_acc_top5, test_stats["top5"])

        session.report({"loss": test_stats["loss"], "accuracy": test_stats["top1"]})

        if args.rank == 0:
            log_data_and_state_dicts(
                args=args,
                model=model,
                model_paths_dict=model_paths_dict if bool(args.save_model) else None,
                tr_stats=train_stats,
                te_stats=test_stats,
                lr=optimizer.param_groups[0]["lr"],
                epoch=epoch,
                best_top1_acc=best_acc_top1,
                best_top5_acc=best_acc_top5,
                best_acc_top1_prev=best_acc_top1_prev,
            )

    if args.rank == 0:
        wandb.finish()

    torch.cuda.empty_cache()
    torch.distributed.destroy_process_group()
