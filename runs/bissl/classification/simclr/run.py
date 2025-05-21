from runs.distributed import init_dist
from runs.bissl.classification.run_contrastive_default import run_contrastive
from runs.bissl.classification.simclr.config import ArgsBiSSLClassificationSimCLR
from runs.misc import override_pretext_config
from runs.pretext.simclr.config import ArgsPretextSimCLR
from source.models.simclr import SimCLR

if __name__ == "__main__":
    args = ArgsBiSSLClassificationSimCLR().parse_args()

    ### Distributed Setup ###
    device = init_dist(args)

    pretrain_config = ArgsPretextSimCLR()
    override_pretext_config(args, pretrain_config)
    assert pretrain_config.pretext_task == "simclr"

    model_p = SimCLR(
        backbone_arch=pretrain_config.backbone_arch,
        proj_mlp=pretrain_config.projector_mlp,
        temp=args.p_simclr_temperature or pretrain_config.temperature,
        distributed=True,
    )

    run_contrastive(
        args=args,
        pretrain_config=pretrain_config,
        device=device,
        model_p=model_p,
    )
