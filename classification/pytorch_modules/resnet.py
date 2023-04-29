import hydra
import torch
import torch.nn as nn
from omegaconf import DictConfig


def get_backbone(registry: str, name: str) -> nn.Sequential:
    backbone = nn.Sequential(
        *list(torch.hub.load(registry, name).children())[:8]
    )
    num_features = 2048
    return backbone, num_features

def get_head(num_features: int, num_layers: int) -> nn.Sequential:
    fastai_head = nn.Sequential(
        nn.AdaptiveAvgPool2d((1, 1)),
        nn.Flatten(),
        nn.BatchNorm1d(num_features),
        nn.Dropout(p=0.25),
        nn.Linear(num_features, out_features=512, bias=False),
        nn.ReLU(),
        nn.BatchNorm1d(num_features=512),
        nn.Dropout(),
        nn.Linear(in_features=512, out_features=num_layers),
    )

    # Source : https://docs.fast.ai/vision.learner.html
    
    return fastai_head

class Classifier(nn.Module):
    def __init__(self, registry: str, name: str, num_layers: int = 2):
        super().__init__()
        self.backbone, num_features = get_backbone(registry, name)
        self.head = get_head(num_features=num_features, num_layers=num_layers)

    def forward(self, x):
        features = self.backbone(x)
        return self.head(features)
