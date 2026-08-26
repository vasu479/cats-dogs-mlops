"""M3 - unit tests for the data pre-processing functions.

Covers ``preprocess_image`` (the 224x224 RGB normalisation every image passes
through) and ``stratified_split`` (the deterministic 80/10/10 split).
"""

from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from src.data_prep import preprocess_image, stratified_split


# ---------------------------------------------------------------------------
# preprocess_image
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "mode,size",
    [
        ("RGB", (500, 300)),    # landscape colour
        ("RGB", (120, 640)),    # portrait colour
        ("L", (64, 64)),        # grayscale -> must become 3-channel
        ("RGBA", (300, 300)),   # alpha channel -> must be dropped
        ("P", (200, 100)),      # palette image
    ],
)
def test_preprocess_image_always_returns_224_rgb(mode: str, size: tuple) -> None:
    image = Image.new(mode, size)
    result = preprocess_image(image, image_size=224)

    assert result.mode == "RGB", "model input must always have 3 channels"
    assert result.size == (224, 224), "model input must always be 224x224"


def test_preprocess_image_respects_custom_size() -> None:
    result = preprocess_image(Image.new("RGB", (400, 400)), image_size=64)
    assert result.size == (64, 64)


def test_preprocess_image_is_a_no_op_for_already_conforming_input() -> None:
    original = Image.new("RGB", (224, 224), color=(10, 20, 30))
    result = preprocess_image(original, image_size=224)
    assert result.size == (224, 224)
    assert result.getpixel((0, 0)) == (10, 20, 30), "no resample should have occurred"


@pytest.mark.parametrize("bad_size", [0, -1, 224.0, "224", None])
def test_preprocess_image_rejects_invalid_size(bad_size) -> None:
    with pytest.raises(ValueError):
        preprocess_image(Image.new("RGB", (50, 50)), image_size=bad_size)


# ---------------------------------------------------------------------------
# stratified_split
# ---------------------------------------------------------------------------
@pytest.fixture()
def sample_paths() -> list:
    return [Path(f"img_{i:04d}.jpg") for i in range(100)]


def test_split_ratios_are_respected(sample_paths: list) -> None:
    splits = stratified_split(sample_paths, 0.8, 0.1, 0.1, seed=42)
    assert len(splits["train"]) == 80
    assert len(splits["val"]) == 10
    assert len(splits["test"]) == 10


def test_split_is_exhaustive_and_disjoint(sample_paths: list) -> None:
    splits = stratified_split(sample_paths, 0.8, 0.1, 0.1, seed=42)
    combined = splits["train"] + splits["val"] + splits["test"]

    assert len(combined) == len(sample_paths), "no image may be dropped"
    assert len(set(combined)) == len(sample_paths), "no image may appear twice"
    assert set(combined) == set(sample_paths)


def test_split_is_deterministic_for_a_fixed_seed(sample_paths: list) -> None:
    first = stratified_split(sample_paths, 0.8, 0.1, 0.1, seed=42)
    second = stratified_split(sample_paths, 0.8, 0.1, 0.1, seed=42)
    assert first == second, "same seed must reproduce the same split"


def test_split_changes_with_a_different_seed(sample_paths: list) -> None:
    first = stratified_split(sample_paths, 0.8, 0.1, 0.1, seed=1)
    second = stratified_split(sample_paths, 0.8, 0.1, 0.1, seed=2)
    assert first["train"] != second["train"]


def test_split_is_insensitive_to_input_ordering(sample_paths: list) -> None:
    shuffled = list(reversed(sample_paths))
    assert stratified_split(sample_paths, 0.8, 0.1, 0.1, 7) == stratified_split(
        shuffled, 0.8, 0.1, 0.1, 7
    )


def test_split_rejects_ratios_that_do_not_sum_to_one(sample_paths: list) -> None:
    with pytest.raises(ValueError, match="sum to 1.0"):
        stratified_split(sample_paths, 0.8, 0.1, 0.2, seed=42)


def test_split_handles_a_tiny_collection() -> None:
    splits = stratified_split([Path("a.jpg"), Path("b.jpg")], 0.8, 0.1, 0.1, seed=0)
    assert sum(len(v) for v in splits.values()) == 2
