from runs.distributed import init_dist
from runs.bissl.object_detection.run_contrastive_default import run_contrastive
from runs.bissl.object_detection.byol.config import ArgsBiSSLDetectionBYOL
from runs.pretext.byol.config import ArgsPretextBYOL
from source.models.byol import BYOL
from runs.misc import override_pretext_config

if __name__ == "__main__":
    args = ArgsBiSSLDetectionBYOL().parse_args()

    ### Distributed Setup ###
    device = init_dist(args)

    pretrain_config = ArgsPretextBYOL()
    override_pretext_config(args, pretrain_config)
    assert pretrain_config.pretext_task == "byol"

    model_p = BYOL(
        backbone_arch=pretrain_config.backbone_arch,
        proj_mlp=pretrain_config.projector_mlp,
        distributed=True,
        moving_average_decay=args.p_byol_ma_decay or pretrain_config.ma_decay,
        ma_use_scheduler=bool(
            args.p_byol_ma_use_scheduler or pretrain_config.ma_use_scheduler
        ),
        ma_scheduler_length=args.lower_num_iter * args.epochs,
    )

    run_contrastive(
        args=args,
        pretrain_config=pretrain_config,
        device=device,
        model_p=model_p,
        hinv_solver_lower_criterion=lambda input_pair: model_p.forward(
            input_pair, update_ema=False
        ),
    )
