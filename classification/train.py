import os
import hydra
import comet_ml
import torch
import pandas as pd
from utilitaries.utils import flatten_dict
import pytorch_lightning as pl
from omegaconf import DictConfig
from pytorch_lightning.loggers.comet import CometLogger
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_modules.resnet import Classifier
from lightning_modules.resnet import ClassificationModel
from lightning_modules.datasets import ClassificationData


torch.backends.cudnn.benchmark = True

@hydra.main(config_path="configs/", config_name="config.yaml")
def main(cfg: DictConfig):
    experiment_name = cfg.experiment.name

    # 2 : Instantiate Comet Logger
    logger = CometLogger(
        project_name=cfg.comet.project,
        experiment_name=experiment_name,
        api_key=cfg.comet.key,
        workspace=cfg.comet.workspace,
    )
    hparams = flatten_dict(cfg)
    logger.log_hyperparams(hparams)

    # 3 : Data Module
    data_module = ClassificationData(
        df=pd.read_csv(cfg.data.data_path, index_col=0),
        batch_size=cfg.experiment.batch_size,
        num_workers=2,
        pin_memory=True,
    )

    # 4 : Model & criterion
    model = Classifier(**cfg.model)
    criterion = torch.nn.CrossEntropyLoss()

    # 5 : Optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg.experiment.learning_rate,
        betas=(0.9, 0.999),
        eps=1e-08,
        weight_decay=cfg.experiment.weight_decay,
        amsgrad=True,
    )

    # 6 : Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer=optimizer,
        patience=3,
        verbose=True,
        mode="min",
        factor=0.1,
        min_lr=1e-08,
    )

    # 7 : Lightning Module for the model
    model_module = ClassificationModel(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        unfreeze_at_beginning=cfg.experiment.unfreeze_at_beginning,
        unfreeze_at_epoch=cfg.experiment.unfreeze_at_epoch,
    )

    # 8 : Callbacks
    callbacks = ModelCheckpoint(
        dirpath=cfg.experiment.checkpoint_dir,
        filename=f"best-validation-loss",
        save_top_k=1,
        monitor=f"Validation-Loss",
        mode="min",
    )

    # 9 : Trainer
    trainer = pl.Trainer(
        logger=logger,
        callbacks=callbacks,
        accelerator='gpu', 
        devices=[0],
    )

    trainer.fit(model=model_module, datamodule=data_module)

if __name__ == "__main__":
    main()
