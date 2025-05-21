from functools import partial
from typing import get_args

from source.types import DetectionBackbones

from runs.distributed import init_dist
from runs.fine_tune.hpo_ray import start_hpo
from runs.fine_tune.misc import (
    get_pretext_and_bissl_configs,
    get_pretrain_config_dict,
    update_args_postbissl,
)

from runs.fine_tune.object_detection.resnet.post_bissl_ft.config import (
    ArgsFTDetectionPostBiSSL,
)
from runs.fine_tune.object_detection.resnet.train import train_detection_resnet

if __name__ == "__main__":
    args = ArgsFTDetectionPostBiSSL().parse_args()

    # Distributed Setup
    _ = init_dist(args, init_process_group=False)

    # Get pretraining configs, updated with the wandb run config.
    pretext_config, bissl_config = get_pretext_and_bissl_configs(args)
    # Override the parsed args with the BiSSL config for specified post-bissl hyperparameters
    # that are None. Non-None hyperparameters are unchanged.

    update_args_postbissl(args, bissl_config, pretext_config)

    assert pretext_config.backbone_arch in get_args(DetectionBackbones)

    # The main trainer script.
    trainer_fn = partial(
        train_detection_resnet,
        args=args,
        backbone_arch=pretext_config.backbone_arch,  # Already asserted the backbone is compatible # type: ignore
        pt_config_dict=get_pretrain_config_dict(pretext_config, bissl_config),
        seed=args.seed,
    )

    # Start the HPO process, using the trainer function.
    start_hpo(args=args, train_fn=trainer_fn)
