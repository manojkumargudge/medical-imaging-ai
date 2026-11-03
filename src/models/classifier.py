"""DenseNet121 with a 14-output multi-label head (one logit per finding, trained with BCE)."""

from __future__ import annotations

from torch import nn
from torchvision.models import DenseNet121_Weights, densenet121


def build_model(num_labels: int, weights: str | None) -> nn.Module:
    """weights: a torchvision DenseNet121_Weights name such as "IMAGENET1K_V1", or None."""
    model = densenet121(weights=DenseNet121_Weights[weights] if weights else None)
    in_features = model.classifier.in_features
    model.classifier = nn.Linear(in_features, num_labels)
    return model  # type: ignore[no-any-return]


def build_loss() -> nn.Module:
    # Sigmoid + BCE per label: the 14 findings are independent yes/no questions.
    return nn.BCEWithLogitsLoss()
