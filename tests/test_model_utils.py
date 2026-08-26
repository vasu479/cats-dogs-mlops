"""M3 - unit tests for the model / inference utilities.

Covers ``bytes_to_tensor`` (upload -> model input), ``probabilities_to_response``
(model output -> API payload), and the model factory.
"""

from __future__ import annotations

import io

import pytest
import torch
from PIL import Image

from src.inference import (
    InvalidImageError,
    bytes_to_tensor,
    probabilities_to_response,
)
from src.model import SimpleCNN, build_model, count_parameters


def make_image_bytes(
    size: tuple = (320, 240), mode: str = "RGB", fmt: str = "JPEG"
) -> bytes:
    buffer = io.BytesIO()
    Image.new(mode, size, color=(120, 80, 200) if mode == "RGB" else 128).save(
        buffer, format=fmt
    )
    return buffer.getvalue()


# ---------------------------------------------------------------------------
# bytes_to_tensor
# ---------------------------------------------------------------------------
def test_bytes_to_tensor_produces_a_batched_3x224x224_input() -> None:
    tensor = bytes_to_tensor(make_image_bytes(), image_size=224)
    assert tensor.shape == (1, 3, 224, 224)
    assert tensor.dtype == torch.float32


@pytest.mark.parametrize(
    "size,mode,fmt",
    [
        ((64, 512), "RGB", "PNG"),
        ((1024, 40), "RGB", "JPEG"),
        ((100, 100), "L", "PNG"),     # grayscale must be widened to 3 channels
    ],
)
def test_bytes_to_tensor_normalises_any_shape_or_mode(size, mode, fmt) -> None:
    assert bytes_to_tensor(make_image_bytes(size, mode, fmt)).shape == (1, 3, 224, 224)


def test_bytes_to_tensor_applies_normalisation() -> None:
    """After ImageNet normalisation, values must leave the raw [0, 1] range."""
    tensor = bytes_to_tensor(make_image_bytes())
    assert tensor.min() < 0.0, "normalised tensors contain negative values"


def test_bytes_to_tensor_rejects_empty_payload() -> None:
    with pytest.raises(InvalidImageError, match="Empty upload"):
        bytes_to_tensor(b"")


def test_bytes_to_tensor_rejects_non_image_bytes() -> None:
    with pytest.raises(InvalidImageError, match="Could not decode"):
        bytes_to_tensor(b"this is definitely not a JPEG file")


def test_bytes_to_tensor_rejects_oversized_payload() -> None:
    with pytest.raises(InvalidImageError, match="limit"):
        bytes_to_tensor(b"\x00" * (11 * 1024 * 1024))


# ---------------------------------------------------------------------------
# probabilities_to_response
# ---------------------------------------------------------------------------
def test_probabilities_to_response_picks_the_argmax_label() -> None:
    payload = probabilities_to_response(torch.tensor([0.12, 0.88]), ["cat", "dog"])
    assert payload["label"] == "dog"
    assert payload["label_index"] == 1
    assert payload["confidence"] == pytest.approx(0.88, abs=1e-6)


def test_probabilities_to_response_returns_every_class() -> None:
    payload = probabilities_to_response(torch.tensor([0.7, 0.3]), ["cat", "dog"])
    assert set(payload["probabilities"]) == {"cat", "dog"}
    assert sum(payload["probabilities"].values()) == pytest.approx(1.0, abs=1e-5)


def test_probabilities_to_response_accepts_a_batched_row() -> None:
    payload = probabilities_to_response(torch.tensor([[0.2, 0.8]]), ["cat", "dog"])
    assert payload["label"] == "dog"


def test_probabilities_to_response_rejects_a_class_count_mismatch() -> None:
    with pytest.raises(ValueError, match="class names"):
        probabilities_to_response(torch.tensor([0.3, 0.3, 0.4]), ["cat", "dog"])


# ---------------------------------------------------------------------------
# model factory
# ---------------------------------------------------------------------------
def test_build_model_forward_pass_shape() -> None:
    model = build_model("simple_cnn", num_classes=2, dropout=0.3).eval()
    with torch.no_grad():
        logits = model(torch.randn(4, 3, 224, 224))
    assert logits.shape == (4, 2)


def test_build_model_output_is_a_valid_distribution_after_softmax() -> None:
    model = build_model().eval()
    with torch.no_grad():
        probs = torch.softmax(model(torch.randn(2, 3, 224, 224)), dim=1)
    assert torch.allclose(probs.sum(dim=1), torch.ones(2), atol=1e-5)
    assert bool((probs >= 0).all())


def test_build_model_rejects_an_unknown_architecture() -> None:
    with pytest.raises(ValueError, match="Unknown model name"):
        build_model("resnet152")


def test_count_parameters_reports_trainable_weights() -> None:
    total, trainable = count_parameters(SimpleCNN())
    assert total > 0
    assert trainable == total, "no layer is frozen in the baseline"


def test_model_state_dict_round_trips(tmp_path) -> None:
    """A saved checkpoint must reload into an identical model."""
    original = build_model().eval()
    checkpoint_path = tmp_path / "model.pt"
    torch.save({"state_dict": original.state_dict()}, checkpoint_path)

    restored = build_model().eval()
    restored.load_state_dict(
        torch.load(checkpoint_path, map_location="cpu", weights_only=False)["state_dict"]
    )

    sample = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        assert torch.allclose(original(sample), restored(sample), atol=1e-6)
