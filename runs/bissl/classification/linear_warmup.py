import torch
from source.trainers import train_classifier
from source.evals import (
    eval_classifier_top1_and_top5,
    eval_classifier_mAP,
)

from runs.bissl.classification.config import ArgsBiSSLClassification
from runs.bissl.linear_warmup import LinearWarmupTrainer


class ClassifierLinearWarmupTrainer(LinearWarmupTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss_fn = torch.nn.CrossEntropyLoss()

    def train_one_epoch(self):
        return train_classifier(
            model=self.model,
            loss_fn=self.loss_fn,
            dataloader=self.dataloader_train,
            optimizer=self.optimizer,
            device=self.device,
        )

    def eval_fn(self, args: ArgsBiSSLClassification):
        if self.dataloader_val is None:
            return 0, 0, 0
        elif args.d_dataset == "voc07":
            return (
                eval_classifier_mAP(
                    model=self.model,
                    dataloader=self.dataloader_val,
                    device=self.device,
                    label="Test Performance (LW, mAP)",
                ),
                0,
                0,
            )
        return eval_classifier_top1_and_top5(
            model=self.model,
            dataloader=self.dataloader_val,
            loss_fn=self.loss_fn,
            device=self.device,
            label="Test Performance (LW)",
        )
