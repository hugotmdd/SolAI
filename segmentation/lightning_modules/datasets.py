import pytorch_lightning as pl
from typing import Union, List
from utils import transforms as T
from torch.utils.data import DataLoader
from pytorch_modules.datasets import SegmentationDataset


class SegmentationData(pl.LightningDataModule):
    def __init__(
        self,
        train_csv: str,
        val_csv: str,
        batch_size: int,
        num_workers: int,
        pin_memory: bool = True,
    ):
        super(SegmentationData, self).__init__()
        self.transforms = T.Compose([T.ToTensor(), T.Normalize()])
        self.train_set = SegmentationDataset(train_csv, transforms=self.transforms)
        self.val_set = SegmentationDataset(val_csv, transforms=self.transforms, train=False)
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