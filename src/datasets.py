"""Torch datasets and augmentation transforms for the processed image folders.

Augmentation (assignment M1: "use data augmentation for better generalization")
is applied to the training split only; validation and test use the deterministic
resize + normalise path so metrics stay comparable across runs.
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# ImageNet channel statistics - standard for 3-channel 224x224 CNN inputs.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_train_transform(image_size: int = 224) -> transforms.Compose:
    """Augmented pipeline: random crop, flip, rotation and colour jitter."""
    return transforms.Compose(
        [
            transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def build_eval_transform(image_size: int = 224) -> transforms.Compose:
    """Deterministic pipeline used for validation, test and live inference."""
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def build_dataloaders(
    processed_dir: Path,
    image_size: int,
    batch_size: int,
    num_workers: int = 0,
) -> Tuple[DataLoader, DataLoader, DataLoader, list]:
    """Create train/val/test loaders from ``processed_dir/{split}/{class}``.

    Returns:
        (train_loader, val_loader, test_loader, class_names)
    """
    train_ds = datasets.ImageFolder(
        processed_dir / "train", transform=build_train_transform(image_size)
    )
    val_ds = datasets.ImageFolder(
        processed_dir / "val", transform=build_eval_transform(image_size)
    )
    test_ds = datasets.ImageFolder(
        processed_dir / "test", transform=build_eval_transform(image_size)
    )

    common = {"num_workers": num_workers, "pin_memory": False}
    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True, **common),
        DataLoader(val_ds, batch_size=batch_size, shuffle=False, **common),
        DataLoader(test_ds, batch_size=batch_size, shuffle=False, **common),
        train_ds.classes,
    )
