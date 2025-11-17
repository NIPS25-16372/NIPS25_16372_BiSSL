from typing import Any, Callable, Dict, List, Optional

import torch
from torch.utils.data import DataLoader
from source.schedulers import CosineLRSchedulerWithWarmup, ConstantLR


class PretextTrainerCuda:
    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        input_processor: Optional[Callable[[Any, torch.device], Any]] = None,
        lr_scheduler: Optional[CosineLRSchedulerWithWarmup | ConstantLR] = None,
    ):

        self.model = model
        self.input_processor = (
            (lambda x, device: (x.to(device)))
            if input_processor is None
            else input_processor
        )
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.device = device

    def train_one_epoch(
        self,
        dataloader: DataLoader,
        epoch: int,
        rank: Optional[int] = None,
    ) -> List[Dict[str, float]]:
        loss_avg = 0

        log_dicts = []

        self.model.train()

        for step, data_point in enumerate(dataloader, 1):

            inp = self.input_processor(data_point, self.device)

            loss = self.model(inp)

            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            loss_avg += loss.item()

            self.optimizer.step()

            if rank is None or (rank == 0 and step % (len(dataloader) // 10) == 0):
                print(
                    f"Avg loss over batch no. {step + 1 - (len(dataloader) // 10)}-{step} / {len(dataloader)}: {loss_avg/(len(dataloader) // 10):>5f}"
                )
                pg = self.optimizer.param_groups
                log_dict = {
                    "train_pretext/loss_avg": loss_avg / (len(dataloader) // 10),
                    "train_pretext/lr_backbone": pg[0]["lr"],
                    "train_pretext/epoch": epoch,
                }
                log_dicts.append(log_dict)

                loss_avg = 0
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

        return log_dicts


def train_classifier(
    model: torch.nn.Module,
    loss_fn: torch.nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    lr_scheduler: Optional[CosineLRSchedulerWithWarmup | ConstantLR] = None,
):
    loss_avg = 0
    model.train()

    for imgs, labels in dataloader:
        imgs, labels = (
            imgs.to(device, non_blocking=True),
            labels.to(device, non_blocking=True),
        )

        # with torch.cuda.amp.autocast(): # type : ignore
        out = model(imgs)
        loss = loss_fn(out, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_avg += loss_fn(model(imgs), labels).item()

        if lr_scheduler is not None:
            lr_scheduler.step()

    print(
        f"Avg loss over batch no. 1-{len(dataloader)} / {len(dataloader)}: {loss_avg/len(dataloader):>5f}",
    )
