"""Model loading and inference utilities shared by the API and the CLI.

Keeping this out of ``app/main.py`` means the inference path can be unit tested
(tests/test_model_utils.py) without standing up an HTTP server.
"""

from __future__ import annotations

import io
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F
from PIL import Image, UnidentifiedImageError

from src.datasets import build_eval_transform
from src.model import build_model

LOGGER = logging.getLogger("inference")

MAX_IMAGE_BYTES = 10 * 1024 * 1024  # 10 MB guard against oversized uploads


class InvalidImageError(ValueError):
    """Raised when an upload cannot be decoded as an image."""


# ---------------------------------------------------------------------------
# Pure helpers - unit tested
# ---------------------------------------------------------------------------
def bytes_to_tensor(image_bytes: bytes, image_size: int = 224) -> torch.Tensor:
    """Decode raw upload bytes into a normalised, batched model input.

    Args:
        image_bytes: the raw file content of an uploaded image.
        image_size: edge length the model was trained on.

    Returns:
        A float tensor of shape ``(1, 3, image_size, image_size)``.

    Raises:
        InvalidImageError: if the payload is empty, oversized, or not decodable.
    """
    if not image_bytes:
        raise InvalidImageError("Empty upload: no image bytes received.")
    if len(image_bytes) > MAX_IMAGE_BYTES:
        raise InvalidImageError(
            f"Image exceeds the {MAX_IMAGE_BYTES // (1024 * 1024)} MB limit."
        )

    try:
        with Image.open(io.BytesIO(image_bytes)) as img:
            img.load()
            rgb = img.convert("RGB")
    except (UnidentifiedImageError, OSError) as exc:
        raise InvalidImageError(f"Could not decode upload as an image: {exc}") from exc

    tensor = build_eval_transform(image_size)(rgb)
    return tensor.unsqueeze(0)  # add batch dimension


def probabilities_to_response(
    probabilities: torch.Tensor, class_names: List[str]
) -> Dict[str, object]:
    """Turn a 1-D probability tensor into the API response payload.

    Raises:
        ValueError: if the tensor length does not match ``class_names``.
    """
    probs = probabilities.detach().flatten().tolist()
    if len(probs) != len(class_names):
        raise ValueError(
            f"Model produced {len(probs)} scores but {len(class_names)} class names "
            "are configured."
        )
    best_index = int(max(range(len(probs)), key=probs.__getitem__))
    return {
        "label": class_names[best_index],
        "label_index": best_index,
        "confidence": round(float(probs[best_index]), 6),
        "probabilities": {
            name: round(float(score), 6) for name, score in zip(class_names, probs)
        },
    }


# ---------------------------------------------------------------------------
# Service wrapper
# ---------------------------------------------------------------------------
class ModelService:
    """Loads the serialized checkpoint once and serves predictions from memory."""

    def __init__(self, model_path: Path, fallback_class_names: List[str]) -> None:
        self.model_path = Path(model_path)
        self.fallback_class_names = list(fallback_class_names)
        self.model: torch.nn.Module | None = None
        self.class_names: List[str] = list(fallback_class_names)
        self.image_size: int = 224
        self.metadata: Dict[str, object] = {}
        self.is_ready: bool = False
        self.load_error: str | None = None

    def load(self) -> bool:
        """Load the checkpoint. Returns True on success; never raises."""
        try:
            if not self.model_path.exists():
                raise FileNotFoundError(
                    f"Model artifact not found at '{self.model_path}'. "
                    "Run `python -m src.train` (or restore it with `dvc pull`) first."
                )
            checkpoint = torch.load(
                self.model_path, map_location="cpu", weights_only=False
            )
            self.class_names = list(
                checkpoint.get("class_names") or self.fallback_class_names
            )
            self.image_size = int(checkpoint.get("image_size", 224))
            model = build_model(
                checkpoint.get("model_name", "simple_cnn"),
                int(checkpoint.get("num_classes", len(self.class_names))),
                float(checkpoint.get("dropout", 0.3)),
            )
            model.load_state_dict(checkpoint["state_dict"])
            model.eval()

            self.model = model
            self.metadata = {
                "model_name": checkpoint.get("model_name", "simple_cnn"),
                "image_size": self.image_size,
                "class_names": self.class_names,
                "training_metrics": checkpoint.get("metrics", {}),
                "mlflow_run_id": checkpoint.get("mlflow_run_id"),
            }
            self.is_ready = True
            self.load_error = None
            LOGGER.info("Model loaded from %s", self.model_path)
        except Exception as exc:  # noqa: BLE001 - health endpoint reports the reason
            self.is_ready = False
            self.load_error = str(exc)
            LOGGER.error("Model load failed: %s", exc)
        return self.is_ready

    def predict(self, image_bytes: bytes) -> Tuple[Dict[str, object], int]:
        """Predict a class for one image.

        Returns:
            (payload, inference_time_ms)

        Raises:
            RuntimeError: if the model was never loaded successfully.
            InvalidImageError: if the payload is not a decodable image.
        """
        if not self.is_ready or self.model is None:
            raise RuntimeError(
                f"Model is not loaded: {self.load_error or 'unknown reason'}"
            )

        import time

        started = time.perf_counter()
        tensor = bytes_to_tensor(image_bytes, self.image_size)
        with torch.no_grad():
            logits = self.model(tensor)
            probabilities = F.softmax(logits, dim=1)[0]
        payload = probabilities_to_response(probabilities, self.class_names)
        elapsed_ms = int((time.perf_counter() - started) * 1000)
        return payload, elapsed_ms
