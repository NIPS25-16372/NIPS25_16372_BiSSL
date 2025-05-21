from typing import Callable, Optional
import json
import sys

import torch
import wandb

from source.augmentations.downstream import ToTensorMultiInputs
from source.augmentations.pretext import ContrastivePretextTransform
from source.evals import eval_object_detector_ap50
from source.models.downstream import FasterRCNNR50C4BB

# General experiment specific imports
from runs.dataloader import get_dataloader_and_sampler
from runs.pretext.config import ArgsPretextDefaults

# General BiSSL Imports
from runs.bissl.object_detection.config import ArgsBiSSLDetection
from runs.bissl.bissl_utils import get_hinv_solver
from runs.misc import wandb_watch_models
from runs.bissl.misc import wandb_init
from runs.bissl.model import (
    get_lower_input_processor_tupled,
    generate_model_local_paths,
    load_pretext_pretrained_model,
    wrap_model_for_ddp,
)
from runs.bissl.optimizer import (
    get_lower_pretext_optimizer_and_lr_sched,
    get_upper_downstream_optimizer_and_lr_sched,
)
from runs.bissl.train import log_data_and_state_dicts

# Downstream Task Specific Imports
from runs.bissl.object_detection.train import ObjectDetectionBiSSLTrainer
from runs.bissl.object_detection.linear_warmup import ObjectDetectionLinearWarmupTrainer
from runs.bissl.object_detection.model import (
    get_upper_input_processor,
    pretext_bb_r50_to_r50c4,
    # load_pretrained_backbone_into_downstream,
)


def run_contrastive(
    args: ArgsBiSSLDetection,
    pretrain_config: ArgsPretextDefaults,
    device: torch.device,
    model_p: torch.nn.Module,
    hinv_solver_lower_criterion: Optional[torch.nn.Module | Callable] = None,
):
    """
    General run function to run BiSSL for object detection tasks with contrastive pretext tasks.
    As this is written, this is documented to be compatible with SimCLR and BYOL.
    """

    wandb_init(args, pretrain_config)

    if args.rank == 0:
        assert wandb.run is not None
        model_paths_dict = generate_model_local_paths(
            args, pretrain_config, wandb.run.id
        )

    print("(BiSSL) " + " ".join(sys.argv))

    print(json.dumps(dict(train_type="Console Args", data=" ".join(sys.argv))))
    print("")
    print(f"Device = {device}")
    #### DATA IMPORT ####
    dataloader_d_train, sampler_d_train = get_dataloader_and_sampler(
        args=args,
        dataset_name=args.d_dataset,
        split="train",
        batch_size=args.d_batch_size,
        transform=ToTensorMultiInputs(),
        drop_last=True,
        collate_fn=lambda x: tuple(zip(*x)),
    )
    dataloader_d_val, sampler_d_val = get_dataloader_and_sampler(
        args=args,
        dataset_name=args.d_dataset,
        split="val",
        batch_size=args.d_batch_size,
        transform=ToTensorMultiInputs(),
        collate_fn=lambda x: tuple(zip(*x)),
    )
    n_classes = len(dataloader_d_val.dataset.classes)  # type: ignore

    print("Using downstream dataset: " + args.d_dataset)

    ### MODEL IMPORT ###
    model_p = load_pretext_pretrained_model(args, pretrain_config, model_p)

    # Specifically for FasterRCNNs, we need to redefine which parts of the pretext model that belongs to the backbone and head respectively
    model_p = pretext_bb_r50_to_r50c4(model_p)

    assert pretrain_config.backbone_arch == "resnet50"

    model_d = FasterRCNNR50C4BB(
        pretrain_config.backbone_arch,
        n_classes=n_classes + 1,
        min_size=args.d_img_size_min,
        max_size=args.d_img_size_max,
    )
    # model_d = load_pretrained_backbone_into_downstream(model_d, model_p.backbone)
    model_d.backbone.load_state_dict(model_p.backbone.state_dict())

    ## DDP WRAP ##
    model_p = wrap_model_for_ddp(model_p, device)
    model_d = wrap_model_for_ddp(
        model_d,
        device,
        find_unused_parameters=True if args.d_linear_warmup_epochs > 0 else False,
    )

    # Linear Warmup only makes sense if the backbone is somewhat pretrained
    #### LINEAR WARMUP  ###
    if args.d_linear_warmup_epochs > 0:
        lw_trainer = ObjectDetectionLinearWarmupTrainer(
            args,
            model_d,
            dataloader_d_train,
            sampler_d_train,
            dataloader_d_val,
            sampler_d_val,
            device,
        )

        model_d = lw_trainer.conduct_linear_warmup(args)
        del lw_trainer

        model_d = wrap_model_for_ddp(model_d.module, device)

    # Lets wandb track histogram of model parameters and gradients
    if args.rank == 0:
        wandb_watch_models([model_p, model_d], log_freq=len(dataloader_d_train))

    ### TRAINING SETUP  ###
    pretext_data_transform = ContrastivePretextTransform(
        img_size=args.p_img_size or pretrain_config.img_size,
        min_ratio=args.p_img_crop_min_ratio or pretrain_config.img_crop_min_ratio,
    )
    p_dataset_name = args.p_dataset or pretrain_config.dataset
    dataloader_p, sampler_p = get_dataloader_and_sampler(
        args=args,
        dataset_name=p_dataset_name,
        split="unlabeled" if p_dataset_name == "stl10" else "train",
        batch_size=args.p_batch_size or pretrain_config.batch_size,
        transform=pretext_data_transform,
        drop_last=True,
    )
    optimizer_p, lr_scheduler_p = get_lower_pretext_optimizer_and_lr_sched(
        args,
        pretrain_config,
        parameter_groups=[
            dict(
                params=model_p.parameters(),
                weight_decay=args.p_wd or pretrain_config.wd,
            ),
        ],
    )

    optimizer_d, lr_scheduler_d = get_upper_downstream_optimizer_and_lr_sched(
        args,
        parameter_groups=[dict(params=model_d.parameters(), weight_decay=args.d_wd)],
    )

    bissl_trainer = ObjectDetectionBiSSLTrainer(
        optimizers={"upper": optimizer_d, "lower": optimizer_p},
        models={"upper": model_d, "lower": model_p},
        dataloaders={"upper": dataloader_d_train, "lower": dataloader_p},
        device=device,
        ij_grad_calc=get_hinv_solver(
            args,
            lower_backbone=model_p.backbone,
            lower_criterion=(
                hinv_solver_lower_criterion
                if hinv_solver_lower_criterion
                else model_p.forward
            ),
        ),
        verbose=True if args.rank == 0 else False,
        input_processors={
            "upper": get_upper_input_processor(device),
            "lower": get_lower_input_processor_tupled(device),
        },
        samplers={"upper": sampler_d_train, "lower": sampler_p},
        lr_schedulers={"upper": lr_scheduler_d, "lower": lr_scheduler_p},
        num_iters={"upper": args.upper_num_iter, "lower": args.lower_num_iter},
    )

    print("")
    print("BiSSL Training")

    best_top1_acc = 0

    for epoch in range(args.epochs):
        print("")
        print(f"Epoch {epoch+1} / {args.epochs}\n-------------------------------")

        bissl_trainer.train_one_epoch(args.cg_lam)

        sampler_d_val.set_epoch(epoch)

        print("")

        top1_dbb = eval_object_detector_ap50(
            model_d,
            dataloader_d_val,
            device,
            "Validation Performance",
            args.rank,
        )

        # Using this to see if i should save models
        best_acc_top1_prev = best_top1_acc

        best_top1_acc = max(best_top1_acc, top1_dbb)

        if args.rank == 0:
            log_data_and_state_dicts(
                args=args,
                model_p=model_p,
                model_paths_dict=model_paths_dict,
                top1_dbb=top1_dbb,
                top5_dbb=0,
                loss_dbb=0,
                epoch=epoch,
                best_top1_acc=best_top1_acc,
                best_top5_acc=0,
                best_acc_top1_prev=best_acc_top1_prev,
            )

            print("")

    if args.rank == 0:
        wandb.finish()
