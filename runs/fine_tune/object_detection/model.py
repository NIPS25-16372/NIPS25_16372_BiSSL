import torch

from runs.fine_tune.config import ArgsFineTuningDefaults


def load_pretrained_model_and_ddp_wrap(
    args: ArgsFineTuningDefaults, model: torch.nn.Module, device: torch.device
) -> None:
    pretrained_bb_path = (
        (args.model_root or args.root + "/models")
        + "/"
        + args.pretrained_model_backbone
    )

    unexpected_keys = model.backbone.load_state_dict(
        torch.load(pretrained_bb_path, map_location=device), strict=False
    ).unexpected_keys

    assert not any(
        k[:7] != "layer4." for k in unexpected_keys
    ), f"Expected all unexpected keys to be from layer4, got the following: {unexpected_keys}"

    model.to(device)
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = torch.nn.parallel.DistributedDataParallel(model)

    torch.distributed.barrier()
    model.backbone = model.module.backbone
    model.head = model.module.head
