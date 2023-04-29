import cv2
import numpy as np
import pandas as pd
import albumentations as A
from imageio import imread
from torch.utils.data import Dataset


def augment(image, mask):
    augmentation_pipeline = A.Compose(
        [
            A.RandomBrightnessContrast(p=0.8),
            A.RandomGamma(p=0.8),
            A.Blur(blur_limit=(8, 16), p=0.5),
            A.OpticalDistortion(distort_limit=0.1, shift_limit=0.05, p=1),
        ]
    )
    return augmentation_pipeline(image=image, mask=mask)

class SegmentationDataset(Dataset):
    def __init__(self, csv: str, transforms=None, train: bool = True):
        super(SegmentationDataset, self).__init__()
        self.transforms = transforms
        self.train = train
        self.df = pd.read_csv(csv)

    def __len__(self):
        return len(self.df)

    def get_image(self, item):
        image = imread(self.df["image"].iloc[item])
        image = cv2.resize(image, dsize=(448, 448), interpolation=cv2.INTER_AREA)
        return np.array(image)

    def get_mask(self, item):
        mask_path = self.df["mask"].iloc[item]
        mask = imread(mask_path)
        mask = cv2.resize(mask, dsize=(448, 448), interpolation=cv2.INTER_AREA)
        return mask

    def __getitem__(self, item):
        image_np = self.get_image(item)
        mask = self.get_mask(item)
        if self.train:
            image_np, mask = augment(image=image_np, mask=mask).values()
        if self.transforms:
            image, mask = self.transforms(image_np, mask)

        return {"image": image, "mask": mask, "item": item, "image_np": image_np}
