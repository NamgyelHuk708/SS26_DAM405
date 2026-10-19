"""In-process, thread-safe metrics for the API (exposed at GET /metrics).

Tracks request count, error count / rate, latency percentiles, predictions per
class and uptime. Uvicorn may serve requests concurrently, so every mutation
must hold a lock.
"""

from __future__ import annotations

import threading
import time
from collections import Counter, deque
from typing import Any

import numpy as np


class MetricsStore:
    """Thread-safe counters and latency samples."""

    def __init__(self, max_latency_samples: int = 10_000) -> None:
        """Initialise counters, a bounded latency buffer, a lock and the start time."""
        self._lock = threading.Lock()
        self._start_time = time.time()
        self.request_count = 0
        self.error_count = 0
        self._latencies: deque[float] = deque(maxlen=max_latency_samples)
        self._predictions: Counter[int] = Counter()
        self._by_endpoint: Counter[str] = Counter()
        self._by_status: Counter[int] = Counter()

    def record_request(self, endpoint: str, status_code: int, latency_ms: float) -> None:
        """Record one finished request (called by the middleware for EVERY request)."""
        with self._lock:
            self.request_count += 1
            self._latencies.append(latency_ms)
            self._by_endpoint[endpoint] += 1
            self._by_status[status_code] += 1
            if status_code >= 400:
                self.error_count += 1

    def record_prediction(self, predicted_class: int) -> None:
        """Count one prediction by class (called by /predict)."""
        with self._lock:
            self._predictions[predicted_class] += 1

    def snapshot(self) -> dict[str, Any]:
        """Return a JSON-serialisable dict of the current metrics."""
        with self._lock:
            requests, errors = self.request_count, self.error_count
            latencies = np.array(self._latencies, dtype=float)
            predictions = dict(self._predictions)
            by_endpoint = dict(self._by_endpoint)
            by_status = dict(self._by_status)

        if latencies.size:
            p50, p95, p99 = np.percentile(latencies, [50, 95, 99])
            latency = {
                "avg": round(float(latencies.mean()), 3),
                "p50": round(float(p50), 3),
                "p95": round(float(p95), 3),
                "p99": round(float(p99), 3),
                "max": round(float(latencies.max()), 3),
                "samples": int(latencies.size),
            }
        else:
            latency = {"avg": None, "p50": None, "p95": None, "p99": None, "max": None, "samples": 0}

        return {
            "request_count": requests,
            "error_count": errors,
            "error_rate": round(errors / requests, 4) if requests else 0.0,
            "latency_ms": latency,
            "predictions_by_class": {
                "<=50K (0)": predictions.get(0, 0),
                ">50K (1)": predictions.get(1, 0),
            },
            "requests_by_endpoint": by_endpoint,
            "responses_by_status": {str(k): v for k, v in sorted(by_status.items())},
            "uptime_seconds": round(time.time() - self._start_time, 1),
        }
