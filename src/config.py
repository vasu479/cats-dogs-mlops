"""Typed access to params.yaml.

Every script and the serving app read configuration from one place so that a
container, a CI runner and a laptop all resolve identical values.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List

import yaml

# Repository root = parent of the directory holding this file.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
PARAMS_PATH = Path(os.getenv("PARAMS_PATH", PROJECT_ROOT / "params.yaml"))


def _load_raw() -> Dict[str, Any]:
    with open(PARAMS_PATH, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


@dataclass(frozen=True)
class DataConfig:
    raw_dir: Path
    processed_dir: Path
    image_size: int
    train_split: float
    val_split: float
    test_split: float
    seed: int
    max_per_class: int


@dataclass(frozen=True)
class TrainConfig:
    epochs: int
    batch_size: int
    learning_rate: float
    weight_decay: float
    num_workers: int
    seed: int


@dataclass(frozen=True)
class ModelConfig:
    name: str
    num_classes: int
    dropout: float


@dataclass(frozen=True)
class MLflowConfig:
    experiment_name: str
    tracking_uri: str
    run_name: str


@dataclass(frozen=True)
class ServingConfig:
    model_path: Path
    class_names: List[str] = field(default_factory=list)


@dataclass(frozen=True)
class Config:
    data: DataConfig
    train: TrainConfig
    model: ModelConfig
    mlflow: MLflowConfig
    serving: ServingConfig


def _abs(value: str) -> Path:
    """Resolve a params.yaml path relative to the repository root."""
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_config() -> Config:
    raw = _load_raw()
    d, t, m, mf, s = (
        raw["data"],
        raw["train"],
        raw["model"],
        raw["mlflow"],
        raw["serving"],
    )
    return Config(
        data=DataConfig(
            raw_dir=_abs(d["raw_dir"]),
            processed_dir=_abs(d["processed_dir"]),
            image_size=int(d["image_size"]),
            train_split=float(d["train_split"]),
            val_split=float(d["val_split"]),
            test_split=float(d["test_split"]),
            seed=int(d["seed"]),
            max_per_class=int(d["max_per_class"]),
        ),
        train=TrainConfig(
            epochs=int(t["epochs"]),
            batch_size=int(t["batch_size"]),
            learning_rate=float(t["learning_rate"]),
            weight_decay=float(t["weight_decay"]),
            num_workers=int(t["num_workers"]),
            seed=int(t["seed"]),
        ),
        model=ModelConfig(
            name=m["name"],
            num_classes=int(m["num_classes"]),
            dropout=float(m["dropout"]),
        ),
        mlflow=MLflowConfig(
            experiment_name=mf["experiment_name"],
            tracking_uri=mf["tracking_uri"],
            run_name=mf["run_name"],
        ),
        serving=ServingConfig(
            model_path=_abs(s["model_path"]),
            class_names=list(s["class_names"]),
        ),
    )


CONFIG = load_config()
