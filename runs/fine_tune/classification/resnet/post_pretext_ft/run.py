from functools import partial
from typing import get_args

from runs.distributed import init_dist
from runs.fine_tune.hpo_ray import start_hpo
from runs.fine_tune.misc import (
    get_pretext_config,
    get_pretrain_config_dict,
)
from runs.fine_tune.classification.resnet.post_pretext_ft.config import (
    ArgsFTClassificationResNetPostPretext,
)
from runs.fine_tune.classification.resnet.train import train_classification_resnet
from source.types import ConvBackbones

if __name__ == "__main__":
    args = ArgsFTClassificationResNetPostPretext().parse_args()

    # Distributed Setup
    _ = init_dist(args, init_process_group=False)

    # Get pretraining config, updated with the wandb run config.
    pretext_config = get_pretext_config(args)

    assert pretext_config.backbone_arch in get_args(ConvBackbones)

    # The main trainer script.
    trainer_fn = partial(
        train_classification_resnet,
        args=args,
        backbone_arch=pretext_config.backbone_arch,  # Already asserted the backbone is compatible # type: ignore
        pt_config_dict=get_pretrain_config_dict(pretext_config),
        seed=args.seed,
    )

    # Start the HPO process, using the trainer function.
    start_hpo(args=args, train_fn=trainer_fn)
