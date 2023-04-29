import numpy as np
import pandas as pd
from PIL import Image
import albumentations as A
from torch.utils.data import Dataset

def augment(image):
    augmentation_pipeline = A.Compose(
        [
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(),
            A.RandomRotate90(p=0.5),
            A.RandomBrightnessContrast(p=0.5),
            A.RandomGamma(p=0.5),
            A.Blur(blur_limit=(2, 5), p=0.5),
            A.OpticalDistortion(),
        ]
    )
    return augmentation_pipeline(image=image)

class ClassificationDataset(Dataset):
    def __init__(
        self,
        df,
        train=True,
        transform=None,
    ):
        super(ClassificationDataset, self).__init__()
        self.df = df
        self.train = train
        self.transform = transform

        if self.train:
            # Split data for training
            self.pos_files = self.df[self.df["label"] == 1]
            self.pos_files = self.pos_files.iloc[
                0 : int((len(self.pos_files) * 4 // 5))
            ]
            self.neg_files = self.df[self.df["label"] == 0]
            self.neg_files = self.neg_files.iloc[
                0 : int((len(self.neg_files) * 4 // 5))
            ]
        else:
            # Split data for validation
            self.pos_files = self.df[self.df["label"] == 1]
            self.pos_files = self.pos_files.iloc[int((len(self.pos_files) * 4 // 5)) :]
            self.neg_files = self.df[self.df["label"] == 0]
            self.neg_files = self.neg_files.iloc[(int(len(self.neg_files) * 4 // 5)) :]

        # Concatenate positive and negative files
        self.df = pd.concat([self.pos_files, self.neg_files], axis=0).reset_index(
            drop=True
        )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        image = Image.open(self.df["image"].values[idx]).convert("RGB")
        label = self.df["label"].values[idx]

        if self.train:
            # Augment the trainning data
            npimage = np.array(image)
            npimage = augment(npimage)["image"]
            image = Image.fromarray(npimage)

        if self.transform:
            # Apply the transform
            image = self.transform(image)

        return {"image": image, "y": label}
        