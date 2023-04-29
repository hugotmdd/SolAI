import torch
import numpy as np
import pytorch_lightning as pl
from typing import List, Any


class SegmentationModel(pl.LightningModule):
    def __init__(
        self,
        model: torch.nn.Module,
        criterion: torch.nn.Module,
        metric: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler._LRScheduler,
    ):
        super(SegmentationModel, self).__init__()

        self.model = model
        self.optimizer = optimizer
        self.lr_scheduler = scheduler
        self.criterion = criterion
        self.metric = metric

    def training_step(self, batch, batch_idx) -> torch.Tensor:
        image, label = batch["image"], batch["mask"]
        outputs = self.model(image)
        loss = self.criterion(outputs, label)
        prediction = outputs
        metric = self.metric(prediction, label)
        self.log("Training-Loss", loss, prog_bar=True)
        self.log(f"Training-{self.metric.name}", metric, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx) -> dict:
        image, label = batch["image"], batch["mask"]
        outputs = self.model(image)
        loss = self.criterion(outputs, label)
        if self.model.name == "u2net":
            prediction = outputs[0]
        else:
            prediction = outputs
        metric = self.metric(prediction, label)
        return {"loss": loss, "metric": metric}

    def validation_epoch_end(self, outputs: List[Any]) -> None:
        avg_loss = np.mean([output["loss"].item() for output in outputs])
        avg_metric = np.mean([output["metric"].item() for output in outputs])

        self.log("Validation-Loss", avg_loss.item(), prog_bar=True)
        self.log(f"Validation-{self.metric.name}", avg_metric.item(), prog_bar=True)

    def on_fit_start(self) -> None:
        if self.model.name == "unet":
            for param in self.model.encoder.parameters():
                param.requires_grad = True
            print("Callback fit called")

    def configure_optimizers(self) -> Any:
        if isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            print("Scheduler == ReduceLrOnPlateau")
            return {
                "optimizer": self.optimizer,
                "lr_scheduler": {
                    "scheduler": self.lr_scheduler,
                    "monitor": f"Validation-{self.metric.name}",
                },
            }

        elif self.lr_scheduler is not None:
            return [self.optimizer], [self.lr_scheduler]
        else:
            return self.optimizer
