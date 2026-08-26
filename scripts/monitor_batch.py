"""M5 - post-deployment model performance tracking.

Sends a small batch of labelled requests to the *deployed* service, compares the
returned predictions against the known true labels, and writes a JSON report
containing live accuracy, per-class breakdown, a confusion matrix and latency
percentiles. The live accuracy is compared with the accuracy recorded at
training time so drift is visible immediately.

Sample source, in priority order:
  1. --samples-dir (default data/monitoring_samples/<class>/*.jpg) - a small
     labelled holdout committed to the repo so this runs on a clean CI runner.
  2. data/processed/test/<class>/*.jpg - the real held-out split, when present.

Usage:
    python scripts/monitor_batch.py --base-url http://localhost:8000 --samples 20
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import requests

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CLASS_NAMES = ("cat", "dog")
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


def discover_samples(samples_dir: Path, limit: int, seed: int) -> List[Tuple[Path, str]]:
    """Collect (image_path, true_label) pairs, balanced across classes."""
    candidates: Dict[str, List[Path]] = {}
    for class_name in CLASS_NAMES:
        class_dir = samples_dir / class_name
        if class_dir.is_dir():
            candidates[class_name] = sorted(
                p for p in class_dir.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES
            )

    if not any(candidates.values()):
        raise FileNotFoundError(
            f"No labelled images under '{samples_dir}/<cat|dog>/'. "
            "Run `python scripts/make_monitoring_set.py` after training, or pass "
            "--samples-dir pointing at a folder with cat/ and dog/ subfolders."
        )

    rng = random.Random(seed)
    per_class = max(1, limit // len([c for c in candidates if candidates[c]]))
    selected: List[Tuple[Path, str]] = []
    for class_name, paths in candidates.items():
        picked = paths if len(paths) <= per_class else rng.sample(paths, per_class)
        selected.extend((path, class_name) for path in picked)

    rng.shuffle(selected)
    return selected[:limit]


def send_batch(
    base_url: str, samples: List[Tuple[Path, str]]
) -> Tuple[List[dict], List[float]]:
    records: List[dict] = []
    latencies: List[float] = []

    for index, (path, true_label) in enumerate(samples, start=1):
        started = time.perf_counter()
        try:
            response = requests.post(
                f"{base_url}/predict",
                files={"file": (path.name, path.read_bytes(), "image/jpeg")},
                timeout=30,
            )
            latency_ms = (time.perf_counter() - started) * 1000
            latencies.append(latency_ms)

            if response.status_code != 200:
                records.append(
                    {
                        "file": path.name,
                        "true_label": true_label,
                        "predicted_label": None,
                        "confidence": None,
                        "correct": False,
                        "latency_ms": round(latency_ms, 2),
                        "error": f"HTTP {response.status_code}",
                    }
                )
                continue

            body = response.json()
            predicted = body["label"]
            records.append(
                {
                    "file": path.name,
                    "true_label": true_label,
                    "predicted_label": predicted,
                    "confidence": body["confidence"],
                    "probabilities": body["probabilities"],
                    "correct": predicted == true_label,
                    "latency_ms": round(latency_ms, 2),
                }
            )
        except requests.RequestException as exc:
            records.append(
                {
                    "file": path.name,
                    "true_label": true_label,
                    "predicted_label": None,
                    "correct": False,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )

        print(
            f"  [{index:>3}/{len(samples)}] {path.name:<28} "
            f"true={true_label:<4} pred={records[-1].get('predicted_label')}",
            flush=True,
        )

    return records, latencies


def percentile(values: List[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(int(round((pct / 100) * (len(ordered) - 1))), len(ordered) - 1)
    return round(ordered[idx], 2)


def summarise(records: List[dict], latencies: List[float]) -> dict:
    total = len(records)
    correct = sum(1 for r in records if r["correct"])
    errors = [r for r in records if r.get("error")]

    matrix = {t: {p: 0 for p in CLASS_NAMES} for t in CLASS_NAMES}
    per_class = {c: {"support": 0, "correct": 0} for c in CLASS_NAMES}
    for record in records:
        true_label = record["true_label"]
        predicted = record.get("predicted_label")
        per_class[true_label]["support"] += 1
        if predicted in CLASS_NAMES:
            matrix[true_label][predicted] += 1
            if predicted == true_label:
                per_class[true_label]["correct"] += 1

    return {
        "samples": total,
        "successful_requests": total - len(errors),
        "failed_requests": len(errors),
        "live_accuracy": round(correct / total, 4) if total else 0.0,
        "per_class_recall": {
            c: round(v["correct"] / v["support"], 4) if v["support"] else None
            for c, v in per_class.items()
        },
        "per_class_support": {c: v["support"] for c, v in per_class.items()},
        "confusion_matrix": matrix,
        "latency_ms": {
            "avg": round(sum(latencies) / len(latencies), 2) if latencies else 0.0,
            "p50": percentile(latencies, 50),
            "p95": percentile(latencies, 95),
            "max": round(max(latencies), 2) if latencies else 0.0,
        },
    }


def fetch_training_accuracy(base_url: str) -> float | None:
    try:
        info = requests.get(f"{base_url}/model-info", timeout=10).json()
        return info.get("metadata", {}).get("training_metrics", {}).get("test_accuracy")
    except Exception:  # noqa: BLE001
        return None


def fetch_service_metrics(base_url: str) -> dict:
    try:
        return requests.get(f"{base_url}/metrics/json", timeout=10).json()
    except Exception:  # noqa: BLE001
        return {}


def main() -> int:
    parser = argparse.ArgumentParser(description="Post-deployment monitoring batch.")
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--samples", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--samples-dir",
        default=str(PROJECT_ROOT / "data" / "monitoring_samples"),
        help="Folder containing cat/ and dog/ subfolders of labelled images.",
    )
    parser.add_argument(
        "--output", default=str(PROJECT_ROOT / "reports" / "post_deploy_monitoring.json")
    )
    parser.add_argument(
        "--min-accuracy",
        type=float,
        default=0.0,
        help="Exit non-zero if live accuracy falls below this threshold.",
    )
    args = parser.parse_args()

    base_url = args.base_url.rstrip("/")
    samples_dir = Path(args.samples_dir)

    if not (samples_dir / "cat").is_dir() and (
        PROJECT_ROOT / "data" / "processed" / "test" / "cat"
    ).is_dir():
        samples_dir = PROJECT_ROOT / "data" / "processed" / "test"
        print(f"Falling back to the held-out test split at {samples_dir}")

    print(f"\n=== POST-DEPLOYMENT MONITORING :: {base_url} ===\n")
    samples = discover_samples(samples_dir, args.samples, args.seed)
    print(f"Sending {len(samples)} labelled requests from {samples_dir}\n")

    records, latencies = send_batch(base_url, samples)
    summary = summarise(records, latencies)

    training_accuracy = fetch_training_accuracy(base_url)
    summary["training_test_accuracy"] = training_accuracy
    if training_accuracy is not None:
        delta = summary["live_accuracy"] - training_accuracy
        summary["accuracy_delta_vs_training"] = round(delta, 4)
        summary["drift_flag"] = bool(delta < -0.10)

    report = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "base_url": base_url,
        "summary": summary,
        "service_metrics": fetch_service_metrics(base_url),
        "records": records,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("\n--- SUMMARY ---")
    print(f"  samples            : {summary['samples']}")
    print(f"  live accuracy      : {summary['live_accuracy']:.2%}")
    if training_accuracy is not None:
        print(f"  training accuracy  : {training_accuracy:.2%}")
        print(f"  delta              : {summary['accuracy_delta_vs_training']:+.2%}")
        print(f"  drift flag         : {summary['drift_flag']}")
    print(f"  per-class recall   : {summary['per_class_recall']}")
    print(f"  latency p50 / p95  : {summary['latency_ms']['p50']} / "
          f"{summary['latency_ms']['p95']} ms")
    print(f"  failed requests    : {summary['failed_requests']}")
    print(f"\nReport written to {output_path}\n")

    if summary["failed_requests"] > 0:
        print("ERROR: some monitoring requests failed.", file=sys.stderr)
        return 1
    if summary["live_accuracy"] < args.min_accuracy:
        print(
            f"ERROR: live accuracy {summary['live_accuracy']:.2%} is below the "
            f"{args.min_accuracy:.2%} threshold.",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
