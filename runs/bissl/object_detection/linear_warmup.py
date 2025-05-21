from typing import Tuple

from source.trainers import train_object_detector
from source.evals import eval_object_detector_ap50

from runs.bissl.config import ArgsBiSSLDefaults
from runs.bissl.linear_warmup import LinearWarmupTrainer


class ObjectDetectionLinearWarmupTrainer(LinearWarmupTrainer):
    def train_one_epoch(self):
        return train_object_detector(
            model=self.model,
            dataloader=self.dataloader_train,
            optimizer=self.optimizer,
            device=self.device,
            return_loss_avg_total=True,
        )

    def eval_fn(self, args: ArgsBiSSLDefaults) -> Tuple[float, float, float]:
        if self.dataloader_val is None:
            return 0, 0, 0
        return (
            eval_object_detector_ap50(
                model=self.model,
                dataloader=self.dataloader_val,
                device=self.device,
                label="Test Performance (AP50)",
            ),
            0,
            0,
        )
