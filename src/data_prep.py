"""M1 - data preparation.

Takes the raw Kaggle Cats-vs-Dogs folders, normalises every image to
224x224 RGB, and writes a deterministic 80/10/10 train/val/test split to
``data/processed``. Augmentation transforms for the training split are also
defined here and applied at load time by the DataLoader.

Run:  python -m src.data_prep
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import random
import shutil
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

from PIL import Image, UnidentifiedImageError

from src.config import CONFIG

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s"
)
LOGGER = logging.getLogger("data_prep")

CLASS_NAMES: Tuple[str, ...] = ("cat", "dog")
VALID_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}


# ---------------------------------------------------------------------------
# Pure functions - these are the units covered by tests/test_data_prep.py
# ---------------------------------------------------------------------------
def preprocess_image(image: Image.Image, image_size: int = 224) -> Image.Image:
    """Normalise any PIL image to a square ``image_size`` RGB image.

    Two guarantees the rest of the pipeline depends on:
      * mode is exactly ``RGB`` (grayscale, palette and RGBA inputs are converted,
        so the tensor always has 3 channels);
      * size is exactly ``(image_size, image_size)``.

    Args:
        image: any PIL image.
        image_size: target edge length in pixels.

    Returns:
        A new RGB PIL image of shape (image_size, image_size).

    Raises:
        ValueError: if ``image_size`` is not a positive integer.
    """
    if not isinstance(image_size, int) or image_size <= 0:
        raise ValueError(f"image_size must be a positive int, got {image_size!r}")

    if image.mode != "RGB":
        image = image.convert("RGB")
    if image.size != (image_size, image_size):
        image = image.resize((image_size, image_size), Image.BILINEAR)
    return image


def stratified_split(
    items: Sequence[Path],
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int = 42,
) -> Dict[str, List[Path]]:
    """Shuffle ``items`` deterministically and cut them into three disjoint sets.

    The split is exhaustive (every item lands in exactly one bucket) and the
    remainder after flooring goes to the train bucket, so no image is dropped.

    Raises:
        ValueError: if the three ratios do not sum to 1.0 (1e-6 tolerance).
    """
    total_ratio = train_ratio + val_ratio + test_ratio
    if abs(total_ratio - 1.0) > 1e-6:
        raise ValueError(f"splits must sum to 1.0, got {total_ratio}")

    ordered = sorted(items, key=lambda p: p.name)  # stable regardless of FS order
    rng = random.Random(seed)
    rng.shuffle(ordered)

    n = len(ordered)
    n_val = int(n * val_ratio)
    n_test = int(n * test_ratio)
    n_train = n - n_val - n_test

    return {
        "train": ordered[:n_train],
        "val": ordered[n_train : n_train + n_val],
        "test": ordered[n_train + n_val :],
    }


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------
def discover_images(raw_dir: Path, class_name: str, max_per_class: int) -> List[Path]:
    class_dir = raw_dir / class_name
    if not class_dir.is_dir():
        raise FileNotFoundError(
            f"Expected class folder '{class_dir}'. "
            "Run scripts/download_data.ps1 first, or lay out data/raw/<cat|dog>/*.jpg."
        )
    files = sorted(p for p in class_dir.iterdir() if p.suffix.lower() in VALID_SUFFIXES)
    if max_per_class > 0:
        files = files[:max_per_class]
    if not files:
        raise FileNotFoundError(f"No images found under '{class_dir}'.")
    return files


def build_processed_dataset() -> Dict[str, int]:
    cfg = CONFIG.data
    if cfg.processed_dir.exists():
        LOGGER.info("Clearing previous processed data at %s", cfg.processed_dir)
        shutil.rmtree(cfg.processed_dir)

    manifest_rows: List[Dict[str, str]] = []
    counts: Dict[str, int] = {"train": 0, "val": 0, "test": 0, "corrupt": 0}

    for label_idx, class_name in enumerate(CLASS_NAMES):
        files = discover_images(cfg.raw_dir, class_name, cfg.max_per_class)
        LOGGER.info("Found %d raw images for class '%s'", len(files), class_name)

        splits = stratified_split(
            files, cfg.train_split, cfg.val_split, cfg.test_split, cfg.seed
        )
        for split_name, split_files in splits.items():
            out_dir = cfg.processed_dir / split_name / class_name
            out_dir.mkdir(parents=True, exist_ok=True)
            for src_path in split_files:
                try:
                    with Image.open(src_path) as img:
                        img.load()  # forces decode; truncated files raise here
                        processed = preprocess_image(img, cfg.image_size)
                        dest = out_dir / f"{src_path.stem}.jpg"
                        processed.save(dest, format="JPEG", quality=90)
                except (UnidentifiedImageError, OSError) as exc:
                    # The Kaggle set is known to contain a handful of corrupt files.
                    counts["corrupt"] += 1
                    LOGGER.warning("Skipping unreadable image %s (%s)", src_path, exc)
                    continue

                counts[split_name] += 1
                manifest_rows.append(
                    {
                        "split": split_name,
                        "class_name": class_name,
                        "label": str(label_idx),
                        "path": str(dest.relative_to(cfg.processed_dir)).replace(
                            "\\", "/"
                        ),
                    }
                )

    manifest_path = cfg.processed_dir / "manifest.csv"
    with open(manifest_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["split", "class_name", "label", "path"]
        )
        writer.writeheader()
        writer.writerows(manifest_rows)

    summary_path = cfg.processed_dir / "summary.json"
    summary = {
        "image_size": cfg.image_size,
        "classes": list(CLASS_NAMES),
        "counts": counts,
        "total": sum(counts[s] for s in ("train", "val", "test")),
        "seed": cfg.seed,
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    LOGGER.info("Processed dataset written to %s", cfg.processed_dir)
    LOGGER.info("Split counts: %s", counts)
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess Cats vs Dogs images.")
    parser.parse_args()
    build_processed_dataset()


if __name__ == "__main__":
    main()
