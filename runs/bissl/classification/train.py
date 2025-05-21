from typing import Optional, Tuple
import torch
from torch.utils.data import DataLoader

from source.models.downstream import DSClassifier
from source.bissl.bissl_trainer import GeneralBiSSLTrainer
from source.evals import eval_classifier_top1_and_top5, eval_classifier_mAP

from runs.bissl.classification.config import ArgsBiSSLClassification


class ClassificationBiSSLTrainer(GeneralBiSSLTrainer):
    """
    Trainer for downstream classification tasks using BiSSL.

    This class extends GeneralBiSSLTrainer and is specifically designed for
    downstream classification tasks. It uses a cross-entropy loss function
    and expects the model to be an instance of DSClassifier.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        assert isinstance(self.models["upper"].module, DSClassifier)
        self.loss_fn = torch.nn.CrossEntropyLoss()

    def upper_criterion_ext_backbone(
        self,
        upper_input: Tuple[torch.Tensor, torch.Tensor],
        ext_backbone: Optional[torch.nn.Module] = None,
    ) -> torch.Tensor:

        if ext_backbone is not None:
            return self.loss_fn(
                self.models["upper"].head(ext_backbone(upper_input[0])), upper_input[1]
            )
        return self.loss_fn(self.models["upper"](upper_input[0]), upper_input[1])


def eval_fn(
    args: ArgsBiSSLClassification,
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    loss_fn: Optional[torch.nn.Module] = None,
):
    if args.d_dataset == "voc07":
        top1_dbb = eval_classifier_mAP(
            model,
            dataloader,
            device,
            label="Validation Performance (MAP)",
        )
        top5_dbb, loss_dbb = 0, 0
    else:
        top1_dbb, top5_dbb, loss_dbb = eval_classifier_top1_and_top5(
            model,
            dataloader,
            loss_fn,  # type: ignore
            device,
            label="Validation Performance",
        )

    return top1_dbb, top5_dbb, loss_dbb
