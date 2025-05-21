from typing import Any, Dict, List, Optional, Tuple, get_args
import os
import json
import torch
import wandb

from runs.config_general import ArgsGeneralDefaults

from runs.pretext.config import ArgsPretextDefaults
from runs.pretext.byol.config import ArgsPretextBYOL
from runs.pretext.simclr.config import ArgsPretextSimCLR

from runs.bissl.config import ArgsBiSSLDefaults
from runs.bissl.classification.byol.config import ArgsBiSSLClassificationBYOL
from runs.bissl.classification.simclr.config import ArgsBiSSLClassificationSimCLR
from runs.bissl.object_detection.byol.config import ArgsBiSSLDetectionBYOL
from runs.bissl.object_detection.simclr.config import ArgsBiSSLDetectionSimCLR

from runs.fine_tune.config import ArgsFineTuningDefaults

from runs.misc import override_config
from runs.distributed import set_seed, setup_for_distributed

from source.types import DownstreamTaskTypes, PretextTaskTypes


def init_run(
    args: ArgsFineTuningDefaults, hpo_config: Dict[str, Any], seed: Optional[int] = None
) -> torch.device:

    torch.distributed.init_process_group(
        backend="nccl",
        init_method=args.dist_url,
        world_size=args.world_size,
        rank=args.rank,
    )
    torch.distributed.barrier()

    # Disables all printing, as the HPO will print plenty :)
    setup_for_distributed(False)

    if seed is not None:
        args.seed = seed
    else:
        set_seed(args, force_generate_new_seed=True)

    # HPO Config Import
    for key, val in hpo_config.items():
        if bool(args.use_hpo) or vars(args)[key] is None:
            vars(args)[key] = val

    return torch.device(args.gpu)


def get_pretext_config(
    args: ArgsFineTuningDefaults,
    missing_keys_skip: List[str] = [],
) -> ArgsPretextDefaults:
    assert (
        args.pretrained_model_config is not None
    ), "Pretrained model config is not set."
    assert args.post_pretext_or_bissl == "post-pretext"

    model_root = args.model_root or args.root + "/models"
    config_path = model_root + "/" + args.pretrained_model_config
    assert os.path.isfile(config_path), "Pretrained model config file not found."

    with open(config_path, "r") as fp:
        pt_config_dict = json.load(fp)

    match pt_config_dict["pretext_task"]:
        case "simclr":
            pretext_config = ArgsPretextSimCLR()
        case "byol":
            pretext_config = ArgsPretextBYOL()

    # Manually adds a root key to the pretrain config dict
    # to be able to use the same config for both pretrain and fine-tuning.
    if "root" not in pt_config_dict.keys():
        pt_config_dict["root"] = args.root

    override_config(
        config=pretext_config,
        config_override=pt_config_dict,
        missing_keys_skip=set(
            list(ArgsGeneralDefaults()._get_argument_names())  # pylint: disable=W0212
            + missing_keys_skip
        ),
    )

    return pretext_config


def get_pretext_and_bissl_configs(
    args: ArgsFineTuningDefaults,
    missing_keys_skip_pretext: List[str] = [],
    missing_keys_skip_bissl: List[str] = [],
) -> Tuple[ArgsPretextDefaults, ArgsBiSSLDefaults]:
    assert args.pretrained_model_config is not None
    assert args.post_pretext_or_bissl == "post-bissl"

    model_root = args.model_root or args.root + "/models"

    bissl_config_path = model_root + "/" + args.pretrained_model_config
    assert os.path.isfile(bissl_config_path), "Pretrained model config file not found."

    with open(bissl_config_path, "r") as fp:
        bissl_config_dict = json.load(fp)

    pt_config_dict = bissl_config_dict["PT"]
    del bissl_config_dict["PT"]

    assert pt_config_dict["pretext_task"] in get_args(PretextTaskTypes)
    assert bissl_config_dict["downstream_task"] in get_args(DownstreamTaskTypes)

    match pt_config_dict["pretext_task"]:
        case "simclr":
            pretext_config = ArgsPretextSimCLR()
        case "byol":
            pretext_config = ArgsPretextBYOL()

    # Manually adds a root key to the pretrain config dict
    # to be able to use the same config for both pretrain and fine-tuning.
    if "root" not in bissl_config_dict.keys():
        bissl_config_dict["root"] = args.root
    if "root" not in pt_config_dict.keys():
        pt_config_dict["root"] = args.root

    override_config(
        config=pretext_config,
        config_override=pt_config_dict,
        missing_keys_skip=set(
            list(ArgsGeneralDefaults()._get_argument_names())  # pylint: disable=W0212
            + missing_keys_skip_pretext
        ),
    )

    match bissl_config_dict["downstream_task"]:
        case "classification":
            match pt_config_dict["pretext_task"]:
                case "simclr":
                    bissl_config = ArgsBiSSLClassificationSimCLR()
                case "byol":
                    bissl_config = ArgsBiSSLClassificationBYOL()

        case "object_detection":
            match pt_config_dict["pretext_task"]:
                case "simclr":
                    bissl_config = ArgsBiSSLDetectionSimCLR()
                case "byol":
                    bissl_config = ArgsBiSSLDetectionBYOL()

    override_config(
        bissl_config,
        bissl_config_dict,
        missing_keys_skip=set(
            list(ArgsGeneralDefaults()._get_argument_names())  # pylint: disable=W0212
            + missing_keys_skip_bissl
        ),
    )

    return pretext_config, bissl_config


def update_args_postbissl(
    args: ArgsFineTuningDefaults,
    bissl_config: ArgsBiSSLDefaults,
    pretext_config: ArgsPretextDefaults,
) -> None:
    """Updates the fine-tuning config with the BiSSL config.

    Args:
        args (ArgsFineTuningDefaults): Arguments for the fine-tuning run.
        bissl_config (ArgsBiSSLDefaults): Arguments for the BiSSL run.
    """
    if args.dataset is not None:
        raise ValueError(
            "Dataset is already set, will not override with BiSSL config. "
            + "This is not intended behaviour, and is mainly an experimental feature "
            "that we dont allow by default."
        )

    args.dataset = args.dataset or bissl_config.d_dataset
    args.batch_size = args.batch_size or bissl_config.d_batch_size
    args.optimizer = args.optimizer or bissl_config.d_optimizer
    args.momentum = args.momentum or bissl_config.d_momentum
    args.beta1 = args.beta1 or bissl_config.d_beta1
    args.beta2 = args.beta2 or bissl_config.d_beta2

    if hasattr(args, "img_size"):
        args.img_size = (
            args.img_size or bissl_config.d_img_size or pretext_config.img_size
        )
        assert args.img_size is not None

    if hasattr(args, "img_crop_min_ratio"):
        args.img_crop_min_ratio = (
            args.img_crop_min_ratio
            or bissl_config.d_img_crop_min_ratio
            or pretext_config.img_crop_min_ratio
        )
        assert args.img_crop_min_ratio is not None

    if hasattr(args, "img_size_min"):
        args.img_size_min = args.img_size_min or bissl_config.d_img_size_min
        assert args.img_size_min is not None

    if hasattr(args, "img_size_max"):
        args.img_size_max = args.img_size_max or bissl_config.d_img_size_max
        assert args.img_size_max is not None

    if hasattr(args, "vit_drop_path_rate"):
        args.vit_drop_path_rate = (
            args.vit_drop_path_rate or bissl_config.d_vit_drop_path_rate
        )
        assert args.vit_drop_path_rate is not None

    assert all(
        arg is not None
        for arg in (
            args.batch_size,
            args.optimizer,
            args.momentum,
            args.beta1,
            args.beta2,
        )
    )


def get_pretrain_config_dict(
    pretext_config: ArgsPretextDefaults,
    bissl_config: Optional[ArgsBiSSLDefaults] = None,
) -> Dict[str, Any]:

    pretrain_config_dict = dict(PT=pretext_config.as_dict())
    if bissl_config is not None:
        pretrain_config_dict.update(dict(BiSSL=bissl_config.as_dict()))
    return pretrain_config_dict


def wandb_init(
    args: ArgsFineTuningDefaults,
    pretrain_config_dict_to_log: Dict[str, Any],
) -> None:

    if args.rank == 0:
        wandb.init(
            project=args.project_name,
            config=args.as_dict() | pretrain_config_dict_to_log,
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

        wandb.define_metric("test_ft/epoch")
        wandb.define_metric("test_ft/*", step_metric="test_ft/epoch")
