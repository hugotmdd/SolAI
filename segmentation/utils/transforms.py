import random
import torchvision.transforms as tt
from torchvision.transforms import functional as F


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


class RandomHorizontalFlip(object):
    def __init__(self, prob):
        self.prob = prob

    def __call__(self, image, target):
        if random.random() < self.prob:
            image = image.flip(-1)
            target = target.flip(-1)
        return image, target


class RandomVerticalFlip(object):
    def __init__(self, prob):
        self.prob = prob

    def __call__(self, image, target):
        if random.random() < self.prob:
            image = image.flip(-2)
            target = target.flip(-2)
        return image, target


class Normalize(object):
    def __init__(self):
        self.transform = tt.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    def __call__(self, image, target):
        return self.transform(image), target


class ToTensor(object):
    def __call__(self, image, target):
        image = F.to_tensor(image).float()
        target = F.to_tensor(target).float()
        return image, target