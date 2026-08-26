"""M5 - in-process monitoring counters.

Deliberately dependency-free: the assignment allows "logs, Prometheus, or simple
in-app counters", and an in-app registry keeps the container small while still
exposing real request counts and latency percentiles. The same numbers are
rendered in Prometheus text format at /metrics so the service can be scraped
later without code changes.
"""

from __future__ import annotations

import threading
import time
from collections import Counter, deque
from typing import Deque, Dict, List


class MetricsRegistry:
    """Thread-safe counters and a bounded latency window."""

    def __init__(self, latency_window: int = 500) -> None:
        self._lock = threading.Lock()
        self._started_at = time.time()
        self._requests_total: Counter = Counter()          # keyed by endpoint
        self._responses_total: Counter = Counter()         # keyed by "endpoint:status"
        self._predictions_total: Counter = Counter()       # keyed by predicted label
        self._errors_total: Counter = Counter()            # keyed by error type
        self._latencies_ms: Deque[float] = deque(maxlen=latency_window)
        self._latency_sum_ms: float = 0.0
        self._latency_count: int = 0

    # -- recording ---------------------------------------------------------
    def record_request(self, endpoint: str) -> None:
        with self._lock:
            self._requests_total[endpoint] += 1

    def record_response(self, endpoint: str, status_code: int, latency_ms: float) -> None:
        with self._lock:
            self._responses_total[f"{endpoint}:{status_code}"] += 1
            self._latencies_ms.append(latency_ms)
            self._latency_sum_ms += latency_ms
            self._latency_count += 1

    def record_prediction(self, label: str) -> None:
        with self._lock:
            self._predictions_total[label] += 1

    def record_error(self, error_type: str) -> None:
        with self._lock:
            self._errors_total[error_type] += 1

    # -- reporting ---------------------------------------------------------
    @staticmethod
    def _percentile(values: List[float], pct: float) -> float:
        if not values:
            return 0.0
        ordered = sorted(values)
        index = min(int(round((pct / 100.0) * (len(ordered) - 1))), len(ordered) - 1)
        return ordered[index]

    def snapshot(self) -> Dict[str, object]:
        with self._lock:
            latencies = list(self._latencies_ms)
            total_requests = sum(self._requests_total.values())
            avg = (
                self._latency_sum_ms / self._latency_count
                if self._latency_count
                else 0.0
            )
            return {
                "uptime_seconds": round(time.time() - self._started_at, 2),
                "requests_total": total_requests,
                "requests_by_endpoint": dict(self._requests_total),
                "responses_by_status": dict(self._responses_total),
                "predictions_by_label": dict(self._predictions_total),
                "errors_by_type": dict(self._errors_total),
                "latency_ms": {
                    "count": self._latency_count,
                    "avg": round(avg, 2),
                    "p50": round(self._percentile(latencies, 50), 2),
                    "p95": round(self._percentile(latencies, 95), 2),
                    "p99": round(self._percentile(latencies, 99), 2),
                    "max": round(max(latencies), 2) if latencies else 0.0,
                },
            }

    def prometheus_text(self) -> str:
        """Render the snapshot in Prometheus exposition format."""
        snap = self.snapshot()
        lines: List[str] = [
            "# HELP app_uptime_seconds Seconds since the service started.",
            "# TYPE app_uptime_seconds gauge",
            f"app_uptime_seconds {snap['uptime_seconds']}",
            "# HELP app_requests_total Total requests received, by endpoint.",
            "# TYPE app_requests_total counter",
        ]
        for endpoint, count in snap["requests_by_endpoint"].items():  # type: ignore[union-attr]
            lines.append(f'app_requests_total{{endpoint="{endpoint}"}} {count}')

        lines += [
            "# HELP app_responses_total Responses sent, by endpoint and status code.",
            "# TYPE app_responses_total counter",
        ]
        for key, count in snap["responses_by_status"].items():  # type: ignore[union-attr]
            endpoint, _, status = key.rpartition(":")
            lines.append(
                f'app_responses_total{{endpoint="{endpoint}",status="{status}"}} {count}'
            )

        lines += [
            "# HELP app_predictions_total Predictions made, by predicted label.",
            "# TYPE app_predictions_total counter",
        ]
        for label, count in snap["predictions_by_label"].items():  # type: ignore[union-attr]
            lines.append(f'app_predictions_total{{label="{label}"}} {count}')

        lines += [
            "# HELP app_errors_total Errors, by error type.",
            "# TYPE app_errors_total counter",
        ]
        for error_type, count in snap["errors_by_type"].items():  # type: ignore[union-attr]
            lines.append(f'app_errors_total{{type="{error_type}"}} {count}')

        latency = snap["latency_ms"]  # type: ignore[index]
        lines += [
            "# HELP app_request_latency_ms Request latency in milliseconds.",
            "# TYPE app_request_latency_ms summary",
            f'app_request_latency_ms{{quantile="0.5"}} {latency["p50"]}',
            f'app_request_latency_ms{{quantile="0.95"}} {latency["p95"]}',
            f'app_request_latency_ms{{quantile="0.99"}} {latency["p99"]}',
            f"app_request_latency_ms_count {latency['count']}",
            f"app_request_latency_ms_avg {latency['avg']}",
        ]
        return "\n".join(lines) + "\n"


METRICS = MetricsRegistry()
