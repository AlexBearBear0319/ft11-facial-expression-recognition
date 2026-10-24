"""The three neural baselines used in Option 1."""

from __future__ import annotations

import torch.nn as nn
from torchvision.models import resnet18

MODEL_IDS = ("mlp", "cnn", "resnet18")


class MLP(nn.Module):
    def __init__(self, num_classes: int = 7):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(48 * 48, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes),
        )

    def forward(self, values):
        return self.layers(values)


class SmallCNN(nn.Module):
    def __init__(self, num_classes: int = 7):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(6, 16, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 2 * 2, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes),
        )

    def forward(self, values):
        return self.classifier(self.features(values))


def make_model(model_id: str, num_classes: int = 7):
    if model_id == "mlp":
        return MLP(num_classes)
    if model_id == "cnn":
        return SmallCNN(num_classes)
    if model_id == "resnet18":
        model = resnet18(weights=None)
        model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model
    raise ValueError(f"Unknown model id: {model_id}")


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
