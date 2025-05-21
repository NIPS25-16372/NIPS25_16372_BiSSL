import os
import wandb

from runs.distributed import init_dist
from source.augmentations.pretext import ContrastivePretextTransform
from source.models.simclr import SimCLR
from source.trainers import PretextTrainerCuda

from runs.dataloader import get_dataloader_and_sampler

# Run specific imports
from runs.pretext.misc import wandb_init
from runs.pretext.model import (
    model_to_ddp,
    generate_model_local_paths,
    input_processor_tupled,
)
from runs.pretext.optimizer import get_pretext_optimizer_and_lr_sched
from runs.pretext.train import log_data_and_state_dicts

from runs.pretext.simclr.config import ArgsPretextSimCLR

if __name__ == "__main__":
    args = ArgsPretextSimCLR().parse_args()

    assert args.pretext_task == "simclr"

    ### Distributed Setup ###
    device = init_dist(args)
    print(f"(Classic Pretext) {args.as_dict()}")
    print("")
    print(f"Device = {device}")

    #### DATA IMPORT ####
    dataloader, sampler = get_dataloader_and_sampler(
        args=args,
        dataset_name=args.dataset,
        split="unlabeled" if args.dataset == "stl10" else "train",
        batch_size=args.batch_size,
        transform=ContrastivePretextTransform(
            img_size=args.img_size, min_ratio=args.img_crop_min_ratio
        ),
        drop_last=True,
    )
    ### MODEL IMPORT ###
    model = SimCLR(
        backbone_arch=args.backbone_arch,
        proj_mlp=args.projector_mlp,
        temp=args.temperature,
        distributed=True,
    )
    model = model_to_ddp(model, device=device)

    ### WANDB INIT AND MODEL DIR/NAMING SETUP ###
    if args.rank == 0:
        # Initialize wandb
        wandb_run_id = wandb_init(args, model, log_freq=len(dataloader))

        if args.save_model:
            # Generate model paths and retrieve them
            model_paths_dict = generate_model_local_paths(
                args,
                wandb_run_id,
            )

    ### TRAINING SETUP ###
    optimizer, lr_scheduler = get_pretext_optimizer_and_lr_sched(
        args,
        parameter_groups=[
            dict(
                params=model.parameters(),
                weight_decay=args.wd,
            ),
        ],
        dataloader_len=len(dataloader),
    )

    trainer = PretextTrainerCuda(
        model=model,
        input_processor=input_processor_tupled,
        optimizer=optimizer,
        device=device,
        lr_scheduler=lr_scheduler,
    )

    for epoch in range(args.epochs):
        print("")
        print(f"Epoch {epoch+1} / {args.epochs}\n-------------------------------")
        sampler.set_epoch(epoch)

        log_dicts = trainer.train_one_epoch(dataloader, epoch, rank=args.rank)

        if args.rank == 0:
            log_data_and_state_dicts(
                log_dicts=log_dicts,
                model=model,
                model_paths_dict=model_paths_dict,
                save_model=bool(args.save_model),
            )

        print("")

    if args.rank == 0:
        # If training finished sucesfully, delete optimizer state
        if args.save_model:
            os.remove(model_paths_dict["opt"])

        wandb.finish()
