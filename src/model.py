"""M1 - baseline CNN architecture.

A deliberately small 4-block convolutional network. It is the *baseline* the
assignment asks for: enough capacity to learn cats-vs-dogs meaningfully on CPU,
small enough that a full training run finishes in minutes and the serialized
``.pt`` artifact stays a few megabytes.
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn


class SimpleCNN(nn.Module):
    """Conv-BN-ReLU-MaxPool x4 -> global average pool -> dropout -> linear."""

    def __init__(
        self, num_classes: int = 2, dropout: float = 0.3, in_channels: int = 3
    ) -> None:
        super().__init__()
        self.num_classes = num_classes

        def block(cin: int, cout: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Conv2d(cin, cout, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(cout),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            )

        self.features = nn.Sequential(
            block(in_channels, 32),   # 224 -> 112
            block(32, 64),            # 112 -> 56
            block(64, 128),           # 56  -> 28
            block(128, 128),          # 28  -> 14
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x)
        return self.classifier(x)


def build_model(
    name: str = "simple_cnn", num_classes: int = 2, dropout: float = 0.3
) -> nn.Module:
    """Factory so train.py and the serving layer construct the model identically."""
    if name != "simple_cnn":
        raise ValueError(f"Unknown model name: {name!r}. Supported: 'simple_cnn'.")
    return SimpleCNN(num_classes=num_classes, dropout=dropout)


def count_parameters(model: nn.Module) -> Tuple[int, int]:
    """Return (total_parameters, trainable_parameters)."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable
