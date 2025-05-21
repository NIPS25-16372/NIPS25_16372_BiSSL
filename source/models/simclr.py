from collections import OrderedDict
from typing import Tuple

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn

from source.types import ConvBackbones

from .misc import ConvBackbone, FullGatherLayer, MLPProjector


class NTXent(nn.Module):
    """
    Contrastive loss with distributed data parallel support
    """

    LARGE_NUMBER = 1e9

    def __init__(self, tau=1.0, multiplier=2, distributed=False):
        super().__init__()
        self.tau = tau
        self.multiplier = multiplier
        self.distributed = distributed
        self.norm = 1.0

    def forward(self, z):

        n = z.shape[0]
        assert n % self.multiplier == 0

        z = F.normalize(z, p=2, dim=1) / np.sqrt(self.tau)

        if self.distributed:
            # This modified version assumes that the input z is a concatenated batch from all processes, obtained from
            # using all_gather prior to calling this function.

            # Split "back" into "all_gather format" list as [<proc0>, <proc1>, ...]
            z = torch.chunk(z, dist.get_world_size(), dim=0)

            # split it into [<proc0_aug0>, <proc0_aug1>, ..., <proc0_aug(m-1)>, <proc1_aug(m-1)>, ...]
            z_list = [chunk for x in z for chunk in x.chunk(self.multiplier)]

            # sort it to [<proc0_aug0>, <proc1_aug0>, ...] that simply means [<batch_aug0>, <batch_aug1>, ...] as expected below
            z_sorted = []
            for m in range(self.multiplier):
                for i in range(dist.get_world_size()):
                    z_sorted.append(z_list[i * self.multiplier + m])
            z = torch.cat(z_sorted, dim=0)
            n = z.shape[0]

        logits = z @ z.t()
        logits[np.arange(n), np.arange(n)] = -self.LARGE_NUMBER

        logprob = F.log_softmax(logits, dim=1)

        del logits

        # choose all positive objects for an example, for i it would be (i + k * n/m), where k=0...(m-1)
        m = self.multiplier
        labels = (np.repeat(np.arange(n), m) + np.tile(np.arange(m) * n // m, n)) % n
        # remove labels pointet to itself, i.e. (i, i)
        labels = labels.reshape(n, m)[:, 1:].reshape(-1)

        loss = (
            -logprob[np.repeat(np.arange(n), m - 1), labels].sum()
            / n
            / (m - 1)
            / self.norm
        )

        return loss, 0


class SimCLR(nn.Module):
    def __init__(
        self,
        backbone_arch: ConvBackbones,
        proj_mlp: str,
        temp: float = 1,
        distributed=False,
        proj_use_bn=True,
    ):
        super().__init__()

        self.backbone_arch = backbone_arch
        self.proj_mlp = proj_mlp

        self.backbone, self.num_features = ConvBackbone(backbone_arch)

        assert self.num_features is not None

        # In the case for SimCLR, the head solely consists of the projector
        projector = MLPProjector(proj_mlp, self.num_features, use_bn=proj_use_bn)
        self.head = nn.Sequential(OrderedDict([("projector", projector)]))
        # See BYOL for an example where the head contains multiple modules
        # Keeping this setup for consistent notation, even though does not make a practical difference

        self.distributed = distributed

        self.nt_xent_loss = NTXent(tau=temp, distributed=distributed)

    def forward(
        self,
        input_pair: Tuple[torch.Tensor, torch.Tensor],
    ) -> Tuple[torch.Tensor] | Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:

        x, y = input_pair
        x = self.head(self.backbone(x))
        y = self.head(self.backbone(y))

        z = torch.cat((x, y), dim=0)
        if self.distributed:
            z = torch.cat(FullGatherLayer.apply(z))  # type: ignore
        loss, _ = self.nt_xent_loss(z)

        return loss
