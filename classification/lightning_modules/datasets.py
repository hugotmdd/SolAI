from typing import Union, List
import pytorch_lightning as pl
import torchvision.transforms as T
from torch.utils.data import DataLoader
from pytorch_modules.datasets import ClassificationDataset

class ClassificationData(pl.LightningDataModule):
    def __init__(
        self,
        df,
        batch_size: int,
        num_workers: int,
        pin_memory: bool = True,
    ):
        super(ClassificationData, self).__init__()
        self.transforms = T.Compose([
            T.ToTensor(), 
            T.Resize((448, 448)),
            T.Normalize(
                mean=[0.485, 0.456, 0.406], 
                std=[0.229, 0.224, 0.225]
            )
        ])
        self.train_set = ClassificationDataset(
            df,
            transform=self.transforms,
        )
        self.val_set = ClassificationDataset(
            df,
            transform=self.transforms,
            train=False,
        )
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

    def train_dataloader(self, *args, **kwargs) -> DataLoader:
        return DataLoader(
            dataset=self.train_set,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def val_dataloader(self, *args, **kwargs) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(
            dataset=self.val_set,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )