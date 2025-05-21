import wandb
import torch
from runs.pretext.config import ArgsPretextDefaults


def wandb_init(
    args: ArgsPretextDefaults,
    model: torch.nn.Module,
    log_freq: int = 100,
):

    config_dict = args.as_dict()

    wandb.init(
        project=args.project_name,
        config=config_dict,
        save_code=True if args.code_dir_to_log is not None else False,
        settings=(
            wandb.Settings(code_dir=args.code_dir_to_log)
            if args.code_dir_to_log is not None
            else None
        ),
        dir=args.root + "/wandb",
        # Added for ability to run anonymously and offline
        mode="offline",
        anonymous="must",
    )
    assert wandb.run is not None

    wandb.define_metric("train_pretext/epoch")
    wandb.define_metric("train_pretext/*", step_metric="train_pretext/epoch")

    wandb.watch(model, log="all", log_freq=log_freq, log_graph=True)

    return wandb.run.id
