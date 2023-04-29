import torch
import numpy as np
from typing import List, Any, Union
import pytorch_lightning as pl
from torchmetrics import MetricCollection
from torchmetrics.classification import BinaryAccuracy, BinaryPrecision, BinaryRecall

class ClassificationModel(pl.LightningModule):
    def __init__(
        self,
        model: torch.nn.Module,
        criterion: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler._LRScheduler,
        unfreeze_at_beginning: bool = False,
        unfreeze_at_epoch: Union[int, None] = None,
    ):
        super(ClassificationModel, self).__init__()
        self.criterion = criterion
        self.optimizer = optimizer
        self.lr_scheduler = scheduler
        self.unfreeze_at_beginning = unfreeze_at_beginning
        self.unfreeze_at_epoch = unfreeze_at_epoch
        metrics = MetricCollection([BinaryAccuracy(), BinaryPrecision(), BinaryRecall()])
        self.train_metrics = metrics.clone(prefix="train_")
        self.valid_metrics = metrics.clone(prefix="val_")
        self.model = model

    def training_step(self, batch, batch_idx) -> torch.Tensor:
        image, label = batch["image"], batch["y"].long()
        predictions = self.model(image)
        loss = self.criterion(predictions, label)
        _, predictions = torch.max(predictions, 1)
        outputs = self.train_metrics(predictions, label)
        self.log_dict(outputs, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx) -> dict:
        image, label = batch["image"], batch["y"].long()
        predictions = self.model(image)
        loss = self.criterion(predictions, label)
        _, predictions = torch.max(predictions, 1)
        outputs = self.valid_metrics(predictions, label)
        return {"loss": loss, "metrics": outputs}

    def validation_epoch_end(self, outputs: List[Any]) -> None:
        avg_loss = np.mean([output["loss"].item() for output in outputs])
        avg_acc = np.mean(
            [output["metrics"]["val_BinaryAccuracy"].item() for output in outputs]
        )
        avg_pres = np.mean(
            [output["metrics"]["val_BinaryPrecision"].item() for output in outputs]
        )
        avg_rec = np.mean(
            [output["metrics"]["val_BinaryRecall"].item() for output in outputs]
        )

        self.log("Validation-Loss", avg_loss.item(), prog_bar=True)
        self.log("Validation-Accuracy", avg_acc.item(), prog_bar=True)
        self.log("Validation-Precision", avg_pres.item(), prog_bar=True)
        self.log("Validation-Recall", avg_rec.item(), prog_bar=True)

    def on_fit_start(self) -> None:
        for param in self.model.backbone.parameters():
            param.requires_grad = self.unfreeze_at_beginning
        print(f"Model unfrozen at fit = {self.unfreeze_at_beginning}")

    def on_train_epoch_start(self) -> None:
        if (
            self.unfreeze_at_epoch is not None
            and self.current_epoch == self.unfreeze_at_epoch
        ):
            for param in self.model.backbone.parameters():
                param.requires_grad = True
            if (
                self.unfreeze_at_beginning is False
                and self.current_epoch == self.unfreeze_at_epoch
            ):
                print(f"Model unfrozen at epoch : {self.unfreeze_at_epoch}")

    def configure_optimizers(self) -> Any:
        if isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            print("Scheduler = ReduceLrOnPlateau")
            return {
                "optimizer": self.optimizer,
                "lr_scheduler": {
                    "scheduler": self.lr_scheduler,
                    "monitor": "Validation-Loss",
                },
            }

        elif self.lr_scheduler is not None:
            return [self.optimizer], [self.lr_scheduler]
        else:
            return self.optimizer