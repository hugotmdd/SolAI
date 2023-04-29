import os
import hydra
import torch
import comet_ml
from utils import flatten_dict
from utils.metrics import Dice
from omegaconf import DictConfig
from pytorch_modules.unet import SegModel
from lightning_modules.unet import SegmentationModel
from lightning_modules.datasets import SegmentationData
import pytorch_lightning as pl
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.loggers.comet import CometLogger
from collections import OrderedDict

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
    data_module = SegmentationData(
        train_csv=cfg.data.train_csv,
        val_csv=cfg.data.val_csv,
        batch_size=cfg.experiment.batch_size,
        num_workers=4,
        pin_memory=True,
    )

    # 4 : Model & criterion
    model = SegModel()
    if cfg.experiment.load_segmodel:
        checkpoint = torch.load(cfg.experiment.segmodel_path, map_location="cpu")

        new_checkpoint = OrderedDict()
        for k, v in checkpoint["state_dict"].items():
            name = k[6:]
            new_checkpoint[name] = v
        model.load_state_dict(new_checkpoint)
        optimizer.load_state_dict(checkpoint["optimizer_states"][0])
        print(f"segmodel loaded")

    criterion = torch.nn.BCEWithLogitsLoss()
    metric = Dice(sigmoid=True)

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
        mode="max",
        factor=0.1,
        min_lr=1e-8,
    )

    # 7 : Lightning Module for the model
    model_module = SegmentationModel(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        metric=metric,
    )

    # 8 : Callbacks
    callbacks = ModelCheckpoint(
        dirpath=os.path.join(cfg.experiment.checkpoint_dir, experiment_name),
        filename="best",
        save_top_k=1,
        monitor=f"Validation-{cfg.experiment.metric}",
        mode="max",
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