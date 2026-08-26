"""M1 - training entry point with MLflow experiment tracking.

Logs to MLflow: all hyper-parameters, per-epoch train/val loss and accuracy,
final test metrics, and two artifacts the assignment names explicitly - the
confusion matrix and the loss curves. The serialized model (.pt) is logged as
an MLflow artifact *and* written to models/model.pt for the serving container.

Run:  python -m src.train
"""

from __future__ import annotations

import json
import logging
import random
import time
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")  # headless: required inside Docker and CI runners
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from torch.utils.data import DataLoader

from src.config import CONFIG, PROJECT_ROOT
from src.datasets import build_dataloaders
from src.model import build_model, count_parameters

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s"
)
LOGGER = logging.getLogger("train")

REPORTS_DIR = PROJECT_ROOT / "reports"
MODELS_DIR = PROJECT_ROOT / "models"


def set_seed(seed: int) -> None:
    """Make a run reproducible across Python, NumPy and Torch RNGs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(False)  # keeps cuDNN/CPU kernels available


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
) -> Tuple[float, float]:
    """One pass over ``loader``. Trains when ``optimizer`` is given, else evaluates."""
    is_train = optimizer is not None
    model.train(is_train)

    running_loss, correct, seen = 0.0, 0, 0
    with torch.set_grad_enabled(is_train):
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            if is_train:
                optimizer.zero_grad(set_to_none=True)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            if is_train:
                loss.backward()
                optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            correct += (outputs.argmax(dim=1) == targets).sum().item()
            seen += inputs.size(0)

    return running_loss / max(seen, 1), correct / max(seen, 1)


def collect_predictions(
    model: nn.Module, loader: DataLoader, device: torch.device
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (y_true, y_pred) over an entire loader."""
    model.eval()
    y_true: List[int] = []
    y_pred: List[int] = []
    with torch.no_grad():
        for inputs, targets in loader:
            outputs = model(inputs.to(device))
            y_pred.extend(outputs.argmax(dim=1).cpu().tolist())
            y_true.extend(targets.tolist())
    return np.asarray(y_true), np.asarray(y_pred)


def plot_loss_curves(history: Dict[str, List[float]], out_path: Path) -> Path:
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(11, 4))

    ax_loss.plot(epochs, history["train_loss"], marker="o", label="train")
    ax_loss.plot(epochs, history["val_loss"], marker="o", label="validation")
    ax_loss.set_title("Loss curves")
    ax_loss.set_xlabel("epoch")
    ax_loss.set_ylabel("cross-entropy loss")
    ax_loss.legend()
    ax_loss.grid(alpha=0.3)

    ax_acc.plot(epochs, history["train_acc"], marker="o", label="train")
    ax_acc.plot(epochs, history["val_acc"], marker="o", label="validation")
    ax_acc.set_title("Accuracy curves")
    ax_acc.set_xlabel("epoch")
    ax_acc.set_ylabel("accuracy")
    ax_acc.legend()
    ax_acc.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    return out_path


def plot_confusion_matrix(
    y_true: np.ndarray, y_pred: np.ndarray, class_names: List[str], out_path: Path
) -> Path:
    matrix = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    fig, ax = plt.subplots(figsize=(5, 4.5))
    ConfusionMatrixDisplay(matrix, display_labels=class_names).plot(
        ax=ax, cmap="Blues", colorbar=False
    )
    ax.set_title("Confusion matrix (test split)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    return out_path


def main() -> Dict[str, float]:
    cfg = CONFIG
    set_seed(cfg.train.seed)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    LOGGER.info("Training on device: %s", device)

    train_loader, val_loader, test_loader, class_names = build_dataloaders(
        cfg.data.processed_dir,
        cfg.data.image_size,
        cfg.train.batch_size,
        cfg.train.num_workers,
    )
    LOGGER.info(
        "Dataset sizes -> train=%d val=%d test=%d classes=%s",
        len(train_loader.dataset),
        len(val_loader.dataset),
        len(test_loader.dataset),
        class_names,
    )

    model = build_model(cfg.model.name, cfg.model.num_classes, cfg.model.dropout).to(
        device
    )
    total_params, trainable_params = count_parameters(model)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg.train.learning_rate,
        weight_decay=cfg.train.weight_decay,
    )

    mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)
    mlflow.set_experiment(cfg.mlflow.experiment_name)

    history: Dict[str, List[float]] = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
    }

    with mlflow.start_run(run_name=cfg.mlflow.run_name) as run:
        mlflow.log_params(
            {
                "model_name": cfg.model.name,
                "num_classes": cfg.model.num_classes,
                "dropout": cfg.model.dropout,
                "epochs": cfg.train.epochs,
                "batch_size": cfg.train.batch_size,
                "learning_rate": cfg.train.learning_rate,
                "weight_decay": cfg.train.weight_decay,
                "optimizer": "Adam",
                "image_size": cfg.data.image_size,
                "train_split": cfg.data.train_split,
                "val_split": cfg.data.val_split,
                "test_split": cfg.data.test_split,
                "augmentation": "RandomResizedCrop+HFlip+Rotation15+ColorJitter",
                "seed": cfg.train.seed,
                "total_parameters": total_params,
                "trainable_parameters": trainable_params,
                "train_size": len(train_loader.dataset),
                "val_size": len(val_loader.dataset),
                "test_size": len(test_loader.dataset),
            }
        )

        best_val_acc, best_state = -1.0, None
        start = time.time()
        for epoch in range(1, cfg.train.epochs + 1):
            tr_loss, tr_acc = run_epoch(
                model, train_loader, criterion, optimizer, device
            )
            va_loss, va_acc = run_epoch(model, val_loader, criterion, None, device)

            history["train_loss"].append(tr_loss)
            history["train_acc"].append(tr_acc)
            history["val_loss"].append(va_loss)
            history["val_acc"].append(va_acc)

            mlflow.log_metrics(
                {
                    "train_loss": tr_loss,
                    "train_accuracy": tr_acc,
                    "val_loss": va_loss,
                    "val_accuracy": va_acc,
                },
                step=epoch,
            )
            LOGGER.info(
                "epoch %d/%d | train_loss=%.4f train_acc=%.4f | val_loss=%.4f val_acc=%.4f",
                epoch,
                cfg.train.epochs,
                tr_loss,
                tr_acc,
                va_loss,
                va_acc,
            )

            if va_acc > best_val_acc:
                best_val_acc = va_acc
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        training_seconds = time.time() - start
        if best_state is not None:
            model.load_state_dict(best_state)

        # ---- final evaluation on the held-out test split -------------------
        y_true, y_pred = collect_predictions(model, test_loader, device)
        test_metrics = {
            "test_accuracy": float(accuracy_score(y_true, y_pred)),
            "test_precision": float(precision_score(y_true, y_pred, zero_division=0)),
            "test_recall": float(recall_score(y_true, y_pred, zero_division=0)),
            "test_f1": float(f1_score(y_true, y_pred, zero_division=0)),
            "best_val_accuracy": float(best_val_acc),
            "training_seconds": float(training_seconds),
        }
        mlflow.log_metrics(test_metrics)
        LOGGER.info("Test metrics: %s", json.dumps(test_metrics, indent=2))

        # ---- artifacts -----------------------------------------------------
        cm_path = plot_confusion_matrix(
            y_true, y_pred, class_names, REPORTS_DIR / "confusion_matrix.png"
        )
        curves_path = plot_loss_curves(history, REPORTS_DIR / "loss_curves.png")

        report_txt = classification_report(
            y_true, y_pred, target_names=class_names, zero_division=0
        )
        report_path = REPORTS_DIR / "classification_report.txt"
        report_path.write_text(report_txt, encoding="utf-8")

        metrics_path = REPORTS_DIR / "metrics.json"
        metrics_path.write_text(
            json.dumps({**test_metrics, "history": history}, indent=2), encoding="utf-8"
        )

        checkpoint = {
            "state_dict": model.state_dict(),
            "class_names": class_names,
            "image_size": cfg.data.image_size,
            "model_name": cfg.model.name,
            "num_classes": cfg.model.num_classes,
            "dropout": cfg.model.dropout,
            "metrics": test_metrics,
            "mlflow_run_id": run.info.run_id,
        }
        model_path = MODELS_DIR / "model.pt"
        torch.save(checkpoint, model_path)
        LOGGER.info("Saved model checkpoint -> %s", model_path)

        for artifact in (cm_path, curves_path, report_path, metrics_path, model_path):
            mlflow.log_artifact(str(artifact))

        LOGGER.info("MLflow run_id=%s", run.info.run_id)

    return test_metrics


if __name__ == "__main__":
    main()
