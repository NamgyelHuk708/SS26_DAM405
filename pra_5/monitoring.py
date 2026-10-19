"""
monitoring.py  -  Task 4 (monitoring signals) and Task 5 (drift check)
-----------------------------------------------------------------------
Keeps simple in-memory counters for the service. In production these
numbers would be pushed to a dashboard (Prometheus/Grafana etc.);
here a periodic summary log line is enough.
"""
import logging
from collections import deque

import numpy as np

# Same named logger as the service -> same handlers, same service.log
logger = logging.getLogger("ml-service")

# ---------------------------------------------------------------- Task 4
stats = {
    "requests": 0,                     # total requests received
    "errors": 0,                       # requests that failed
    "latencies": deque(maxlen=100),    # last 100 latencies (ms) only
}


def record(latency_ms, ok=True):
    """Record one request: count it, count errors, store its latency."""
    stats["requests"] += 1
    if not ok:
        stats["errors"] += 1
    stats["latencies"].append(latency_ms)


def summary():
    """Log a one-line health summary and return it as a dict."""
    lat = list(stats["latencies"])
    avg = sum(lat) / len(lat) if lat else 0.0
    p95 = float(np.percentile(lat, 95)) if lat else 0.0
    err_rate = stats["errors"] / stats["requests"] if stats["requests"] else 0.0

    logger.info(f"SUMMARY requests={stats['requests']} "
                f"errors={stats['errors']} error_rate={err_rate:.0%} "
                f"avg_latency_ms={avg:.1f} p95_latency_ms={p95:.1f}")

    return {
        "requests": stats["requests"],
        "errors": stats["errors"],
        "error_rate": round(err_rate, 3),
        "avg_latency_ms": round(avg, 2),
        "p95_latency_ms": round(p95, 2),
    }


# ---------------------------------------------------------------- Task 5
def drift_check(train_mean, live_values, tol=0.25, feature_name="feature"):
    """Compare the live mean of one feature against its training mean.

    Returns the relative change. If it is larger than `tol` (25% by
    default) a WARNING is logged because the input data may have drifted.
    """
    live_mean = float(np.mean(live_values))
    # 1e-9 avoids division by zero if the training mean is 0
    rel = abs(live_mean - train_mean) / (abs(train_mean) + 1e-9)

    if rel > tol:
        logger.warning(f"possible drift [{feature_name}]: "
                       f"live={live_mean:.3f} train={train_mean:.3f} "
                       f"change={rel:.0%}")
    else:
        logger.debug(f"no drift [{feature_name}]: live={live_mean:.3f} "
                     f"train={train_mean:.3f} change={rel:.0%}")
    return rel
