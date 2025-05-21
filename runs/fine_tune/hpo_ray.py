import os
import json
from typing import Any, Callable, Dict

import torch

from ray import tune, init
from ray.tune import TuneConfig

from runs.fine_tune.config import ArgsFineTuningDefaults


def start_hpo(args: ArgsFineTuningDefaults, train_fn: Callable[[Dict[str, Any]], None]):

    # Measures to prevent ray tune from logging too much data.
    # However, it is to the authors knowledge impossible to fully disable the logging of the trial results.
    # Hence, make sure to occasionally clean the ray_spill directory manually...
    os.environ["TUNE_DISABLE_AUTO_CALLBACK_LOGGERS"] = "1"
    os.environ["TUNE_MAX_PENDING_TRIALS_PG"] = str(10 * torch.cuda.device_count())
    os.environ["RAY_AIR_LOCAL_CACHE_DIR"] = args.root + "/ray_spill"

    if args.rank == 0:
        # HPO Setup
        if bool(args.use_hpo):
            assert args.hpo_lr_min is not None and args.hpo_lr_max is not None
            hpo_config = {
                "lr": tune.qloguniform(
                    args.hpo_lr_min, args.hpo_lr_max, args.hpo_lr_min / 100
                ),
                "wd": (
                    tune.choice([args.wd])  # Allow for fixed weight decay
                    if args.hpo_wd_min is None or args.hpo_wd_max is None
                    else tune.qloguniform(
                        args.hpo_wd_min, args.hpo_wd_max, args.hpo_wd_min / 100
                    )
                ),
            }
        else:
            assert (
                args.lr is not None and args.wd is not None
            ), "HPO is not used, but no lr/wd is provided. Please provide --lr and --wd values."
            hpo_config = {"lr": tune.choice([args.lr]), "wd": tune.choice([args.wd])}

        assert all(
            key in args._get_argument_names() for key in hpo_config.keys()
        ), "Some hpo-configurable hyperparameters are not found in the config."

        ngpus_pr_task = 1  # Ray only supports 1 GPU per task.
        args.world_size = (
            ngpus_pr_task  # Will be the world size for each process spawned by ray.
        )
        init(
            num_cpus=args.num_workers * torch.cuda.device_count(),
            num_gpus=torch.cuda.device_count(),
            include_dashboard=False,
            _system_config={
                "local_fs_capacity_threshold": 0.9,
                "min_spilling_size": 100
                * 1024
                * 1024,  # Spill at least 100MB at a time.
                "object_spilling_config": json.dumps(
                    {
                        "type": "filesystem",
                        "params": {
                            "directory_path": args.root + "/ray_spill",
                            "buffer_size": 100 * 1024 * 1024,
                        },
                    },
                ),
            },
        )
        trainable_with_resources = tune.with_resources(
            trainable=train_fn,
            resources={
                "cpu": args.num_workers,
                "gpu": ngpus_pr_task,
            },
        )

        tuner = tune.Tuner(
            trainable_with_resources,
            param_space=hpo_config,
            tune_config=TuneConfig(num_samples=args.num_runs),
        )

        tuner.fit()
