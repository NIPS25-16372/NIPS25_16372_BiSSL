import wandb
from runs.bissl.config import ArgsBiSSLDefaults
from runs.pretext.config import ArgsPretextDefaults


def wandb_init(
    args: ArgsBiSSLDefaults,
    pretrain_config: ArgsPretextDefaults,
) -> None:

    if args.rank == 0:
        wandb.init(
            project=args.project_name,
            config=args.as_dict() | dict(PT=pretrain_config.as_dict()),
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

        # Setting up metrics
        # Linear Warmup
        if args.d_linear_warmup_epochs > 0:
            wandb.define_metric("lw_train_bissl/epoch")
            wandb.define_metric("lw_test/epoch")
            wandb.define_metric("lw_train_bissl/*", step_metric="lw_train_bissl/epoch")
            wandb.define_metric("lw_test/*", step_metric="lw_test/epoch")

        # BiSSL
        wandb.define_metric("train_bissl/epoch")
        wandb.define_metric("test_bisll/epoch")
        wandb.define_metric("train_bissl/*", step_metric="train_bissl/epoch")
        wandb.define_metric("test/*", step_metric="test/epoch")

        wandb.define_metric("top1_acc_test_dbb", step_metric="epoch")
        wandb.define_metric("top5_acc_test_dbb", step_metric="epoch")
        wandb.define_metric("avgloss_test_dbb", step_metric="epoch")

        wandb.define_metric("top1_acc_test_dbb", step_metric="epoch")
        wandb.define_metric("top5_acc_test_dbb", step_metric="epoch")
        wandb.define_metric("avgloss_test_dbb", step_metric="epoch")
