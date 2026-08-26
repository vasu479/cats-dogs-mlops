"""Generate a synthetic stand-in dataset so the whole pipeline can be smoke-tested
without the Kaggle download.

WHY THIS EXISTS: it lets you prove that data prep -> training -> MLflow ->
container -> CI -> CD -> smoke test all work end to end *before* you spend time
on the 800 MB Kaggle dataset. The two synthetic classes are separable, so a
successful run gives high accuracy and confirms the plumbing - it is NOT a
substitute for the real dataset in your submission.

Usage:
    python scripts/make_synthetic_data.py --per-class 240
    python scripts/make_synthetic_data.py --per-class 240 --out data/raw
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def make_cat_like(rng: random.Random, size: int) -> Image.Image:
    """Warm palette, vertical stripes, triangular 'ears'."""
    background = (
        rng.randint(180, 235), rng.randint(140, 195), rng.randint(90, 150)
    )
    image = Image.new("RGB", (size, size), background)
    draw = ImageDraw.Draw(image)

    stripe = (rng.randint(70, 120), rng.randint(45, 85), rng.randint(25, 60))
    for x in range(0, size, rng.randint(12, 20)):
        draw.rectangle([x, 0, x + rng.randint(4, 9), size], fill=stripe)

    cx, cy = size // 2 + rng.randint(-12, 12), size // 2 + rng.randint(-12, 12)
    r = size // 4
    draw.polygon(
        [(cx - r, cy - r), (cx - r // 2, cy - r - r // 2), (cx, cy - r)],
        fill=(rng.randint(200, 255),) * 3,
    )
    draw.polygon(
        [(cx, cy - r), (cx + r // 2, cy - r - r // 2), (cx + r, cy - r)],
        fill=(rng.randint(200, 255),) * 3,
    )
    return image.filter(ImageFilter.GaussianBlur(rng.uniform(0.2, 1.0)))


def make_dog_like(rng: random.Random, size: int) -> Image.Image:
    """Cool palette, horizontal bands, round 'muzzle'."""
    background = (
        rng.randint(70, 130), rng.randint(120, 180), rng.randint(180, 240)
    )
    image = Image.new("RGB", (size, size), background)
    draw = ImageDraw.Draw(image)

    band = (rng.randint(25, 70), rng.randint(60, 110), rng.randint(110, 165))
    for y in range(0, size, rng.randint(12, 20)):
        draw.rectangle([0, y, size, y + rng.randint(4, 9)], fill=band)

    cx, cy = size // 2 + rng.randint(-12, 12), size // 2 + rng.randint(-12, 12)
    r = size // 5
    draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=(rng.randint(15, 60),) * 3)
    draw.ellipse(
        [cx - r // 2, cy - r // 2, cx + r // 2, cy + r // 2],
        fill=(rng.randint(200, 255),) * 3,
    )
    return image.filter(ImageFilter.GaussianBlur(rng.uniform(0.2, 1.0)))


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a synthetic cats/dogs set.")
    parser.add_argument("--per-class", type=int, default=240)
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default=str(PROJECT_ROOT / "data" / "raw"))
    args = parser.parse_args()

    out_root = Path(args.out)
    rng = random.Random(args.seed)

    for class_name, factory in (("cat", make_cat_like), ("dog", make_dog_like)):
        class_dir = out_root / class_name
        class_dir.mkdir(parents=True, exist_ok=True)
        for index in range(args.per_class):
            factory(rng, args.size).save(
                class_dir / f"{class_name}_{index:05d}.jpg", format="JPEG", quality=88
            )
        print(f"Wrote {args.per_class} images -> {class_dir}")

    print(
        "\nSynthetic dataset ready. This validates the pipeline only - replace "
        "data/raw with the real Kaggle Cats-vs-Dogs images before your final run."
    )


if __name__ == "__main__":
    main()
