from typing import Optional, Tuple

import timm
import torch
import torch.nn.functional as F
from torch import nn

from source.types import ConvBackbones


class FullGatherLayer(torch.autograd.Function):
    """
    Gather tensors from all process and support backward propagation
    for the gradients across processes.
    """

    @staticmethod
    def forward(ctx, x: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        output = [
            torch.zeros_like(x) for _ in range(torch.distributed.get_world_size())
        ]
        torch.distributed.all_gather(output, x)
        return tuple(output)

    @staticmethod
    def backward(ctx, *grads):
        all_gradients = torch.stack(grads)
        torch.distributed.all_reduce(all_gradients)
        return all_gradients[torch.distributed.get_rank()]


class MaxPool2dMPS(nn.MaxPool2d):
    """Pytorch only supports achieving second order gradients of maxpool2d on mps devices
    if return_indices=True, thus this minor modification.
    """

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return F.max_pool2d(
            input,
            self.kernel_size,
            self.stride,
            self.padding,
            self.dilation,
            ceil_mode=self.ceil_mode,
            return_indices=True,
        )[0]


def ConvBackbone(
    arch: ConvBackbones,
) -> Tuple[nn.Module, Optional[int]]:

    embedding_dim = None

    if arch.startswith("resnet"):
        mod = timm.create_model(
            arch,
        )
        embedding_dim = mod.fc.in_features
        mod.fc = nn.Flatten()

        # mod.global_pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))

        mod.maxpool = MaxPool2dMPS(
            kernel_size=mod.maxpool.kernel_size,
            stride=mod.maxpool.stride,
            padding=mod.maxpool.padding,
            dilation=mod.maxpool.dilation,
            ceil_mode=mod.maxpool.ceil_mode,
        )

    else:
        raise ValueError(f'Invalid conv backbone architecture "{arch}".')

    return mod, embedding_dim


def MLPProjector(
    arg_mlp: str, embedding_dim: int, use_bn: bool = True
) -> nn.Sequential:
    mlp_spec = f"{embedding_dim}-{arg_mlp}"
    layers = []
    f = list(map(int, mlp_spec.split("-")))
    for i in range(len(f) - 2):
        layers.append(nn.Linear(f[i], f[i + 1]))
        if use_bn:
            layers.append(nn.BatchNorm1d(f[i + 1]))
        layers.append(nn.ReLU(True))
    layers.append(nn.Linear(f[-2], f[-1], bias=False))
    return nn.Sequential(*layers)
