"""Build the small labelled holdout batch used by M5 post-deployment monitoring.

Copies N images per class out of the held-out test split into
``data/monitoring_samples/``. That folder IS committed to Git (it is a few
hundred KB) so the CD job on a clean runner can measure live accuracy against
real labels without needing the full dataset or a DVC remote.

Usage:
    python scripts/make_monitoring_set.py --per-class 10
"""

from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CLASS_NAMES = ("cat", "dog")
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


def main() -> None:
    parser = argparse.ArgumentParser(description="Create the monitoring holdout batch.")
    parser.add_argument("--per-class", type=int, default=10)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--source", default=str(PROJECT_ROOT / "data" / "processed" / "test")
    )
    parser.add_argument(
        "--dest", default=str(PROJECT_ROOT / "data" / "monitoring_samples")
    )
    args = parser.parse_args()

    source = Path(args.source)
    dest = Path(args.dest)
    rng = random.Random(args.seed)

    if not source.is_dir():
        raise SystemExit(
            f"Source split '{source}' not found. Run `python -m src.data_prep` first."
        )

    if dest.exists():
        shutil.rmtree(dest)

    total = 0
    for class_name in CLASS_NAMES:
        class_source = source / class_name
        if not class_source.is_dir():
            raise SystemExit(f"Missing class folder: {class_source}")

        images = sorted(
            p for p in class_source.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES
        )
        picked = images if len(images) <= args.per_class else rng.sample(images, args.per_class)

        class_dest = dest / class_name
        class_dest.mkdir(parents=True, exist_ok=True)
        for path in picked:
            shutil.copy2(path, class_dest / path.name)
        total += len(picked)
        print(f"  {class_name}: copied {len(picked)} images -> {class_dest}")

    print(
        f"\n{total} labelled monitoring images ready in {dest}.\n"
        "Commit this folder so the CD pipeline can measure live accuracy."
    )


if __name__ == "__main__":
    main()
