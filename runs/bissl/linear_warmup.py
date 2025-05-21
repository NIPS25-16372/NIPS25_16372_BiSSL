from typing import Optional, Tuple
import argparse
import wandb
import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from source.optimizers import get_optimizer

from runs.bissl.config import ArgsBiSSLDefaults


class LinearWarmupTrainer:
    def __init__(
        self,
        args: ArgsBiSSLDefaults,
        model: torch.nn.Module,
        dataloader_train: DataLoader,
        sampler_train: Optional[DistributedSampler],
        dataloader_val: Optional[DataLoader],
        sampler_val: Optional[DistributedSampler],
        device: torch.device,
    ):
        self.model = model
        self.dataloader_train = dataloader_train
        self.dataloader_val = dataloader_val
        self.sampler_train = sampler_train
        self.sampler_val = sampler_val
        self.device = device

        self.optimizer = get_optimizer(
            args.d_optimizer,
            params=model.head.parameters(),
            lr=args.d_linear_warmup_lr or args.d_lr,
            wd=args.d_linear_warmup_wd or args.d_wd,
            momentum=args.d_linear_warmup_momentum or args.d_momentum,
            betas=(args.d_beta1, args.d_beta2),
        )

    def train_one_epoch(self) -> float:
        raise NotImplementedError

    def eval_fn(self, args: ArgsBiSSLDefaults) -> Tuple[float, float, float]:
        raise NotImplementedError

    def conduct_linear_warmup(
        self,
        args: ArgsBiSSLDefaults,
    ) -> torch.nn.Module:

        best_acc_lw = argparse.Namespace(top1=0, top5=0)

        for par in self.model.backbone.parameters():
            par.requires_grad = False  # type: ignore

        print("Linear Warmup Training")
        for w_epoch in range(args.d_linear_warmup_epochs):
            print("")
            print(
                f"Warmup Epoch {w_epoch+1} / {args.d_linear_warmup_epochs}\n-------------------------------"
            )

            if self.sampler_train is not None:
                self.sampler_train.set_epoch(
                    w_epoch + args.epochs * args.upper_num_iter + 1
                )
            if self.sampler_val is not None:
                self.sampler_val.set_epoch(
                    w_epoch + args.epochs * args.upper_num_iter + 1
                )

            loss_lw_tr = self.train_one_epoch()

            if (w_epoch + 1) % max((args.d_linear_warmup_epochs // 5), 1) == 0:
                print("")

                top1_lw_tr, top5_lw_tr, loss_lw_tr = self.eval_fn(args)

                best_acc_lw.top1 = max(best_acc_lw.top1, top1_lw_tr)
                best_acc_lw.top5 = max(best_acc_lw.top5, top5_lw_tr)

                if args.rank == 0:
                    wandb.log(
                        {
                            "lw_test/top1_acc_test": top1_lw_tr,
                            "lw_test/top5_acc_test": top5_lw_tr,
                            "lw_test/best_top1_acc_test": best_acc_lw.top1,
                            "lw_test/best_top5_acc_test": best_acc_lw.top5,
                            "lw_test/avgloss_test": loss_lw_tr,
                            "lw_test/epoch": w_epoch,
                        }
                    )
        print("")
        print("FINISHED Linear Warmup Training")
        print("")

        for par in self.model.backbone.parameters():
            par.requires_grad = True  # type: ignore

        return self.model
