"""Data-drift detection: live inputs vs. the training-data reference snapshot.

* Numeric features: two-sample Kolmogorov–Smirnov test; drift if
  ``p_value < drift.ks_p_value_threshold``.
* Categorical features: Population Stability Index; drift if
  ``PSI > drift.psi_threshold``.
* Overall drift if at least ``drift.min_drifted_features`` features drifted.

The same functions are reused offline by ``scripts/offline_drift_report.py``.
"""

from __future__ import annotations

import logging
import threading
from collections import deque
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

from src.config import resolve_path

MISSING = "MISSING"
OTHER = "OTHER"


def ks_test(reference: pd.Series, current: pd.Series) -> tuple[float, float]:
    """Return (KS statistic, p-value) for two numeric samples."""
    ref = pd.to_numeric(reference, errors="coerce").dropna()
    cur = pd.to_numeric(current, errors="coerce").dropna()
    result = ks_2samp(ref, cur)
    return float(result.statistic), float(result.pvalue)


def psi(
    reference: pd.Series, current: pd.Series, min_share: float = 0.0, eps: float = 1e-4
) -> float:
    """Population Stability Index for a categorical feature.

    PSI = sum_over_categories (cur% - ref%) * ln(cur% / ref%)
    Rule of thumb: < 0.1 stable, 0.1–0.2 moderate shift, > 0.2 significant shift.

    Categories whose share in the reference is below ``min_share`` are merged into
    one OTHER bucket. With a small live window, rare categories (e.g. the ~40
    native countries) fluctuate by chance and would otherwise inflate PSI. A
    brand-new category also lands in OTHER, so a real shift is still caught.
    """
    ref = reference.astype(object).where(reference.notna(), MISSING)
    cur = current.astype(object).where(current.notna(), MISSING)
    if min_share > 0:
        shares = ref.value_counts(normalize=True)
        common = set(shares[shares >= min_share].index)
        ref = ref.where(ref.isin(common), OTHER)
        cur = cur.where(cur.isin(common), OTHER)
    categories = sorted(set(ref.unique()) | set(cur.unique()), key=str)
    ref_pct = ref.value_counts(normalize=True).reindex(categories, fill_value=0).to_numpy()
    cur_pct = cur.value_counts(normalize=True).reindex(categories, fill_value=0).to_numpy()
    ref_pct = np.clip(ref_pct, eps, None)
    cur_pct = np.clip(cur_pct, eps, None)
    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))


def compare_frames(
    reference: pd.DataFrame, current: pd.DataFrame, cfg: dict[str, Any]
) -> dict[str, Any]:
    """Compare two frames feature by feature.

    Returns:
        {"drift_detected": bool, "n_drifted": int, "n_samples": int,
         "features": {name: {"type", "test", "statistic", "p_value"?, "threshold", "drifted"}}}
    """
    dcfg = cfg["drift"]
    features: dict[str, dict[str, Any]] = {}

    for col in cfg["data"]["numeric_features"]:
        stat, p_value = ks_test(reference[col], current[col])
        features[col] = {
            "type": "numeric",
            "test": "ks",
            "statistic": round(stat, 4),
            "p_value": round(p_value, 6),
            "threshold": dcfg["ks_p_value_threshold"],
            "drifted": bool(p_value < dcfg["ks_p_value_threshold"]),
        }
    for col in cfg["data"]["categorical_features"]:
        value = psi(reference[col], current[col], min_share=dcfg.get("psi_min_category_share", 0.0))
        features[col] = {
            "type": "categorical",
            "test": "psi",
            "statistic": round(value, 4),
            "threshold": dcfg["psi_threshold"],
            "drifted": bool(value > dcfg["psi_threshold"]),
        }

    drifted = [name for name, f in features.items() if f["drifted"]]
    return {
        "drift_detected": len(drifted) >= dcfg["min_drifted_features"],
        "n_drifted": len(drifted),
        "drifted_features": drifted,
        "min_drifted_features": dcfg["min_drifted_features"],
        "n_samples": int(len(current)),
        "features": features,
    }


class DriftDetector:
    """Keeps a rolling window of live requests and checks it against the reference."""

    def __init__(self, cfg: dict[str, Any], logger: logging.Logger | None = None) -> None:
        """Load the reference parquet and create an empty rolling window."""
        self.cfg = cfg
        self.logger = logger or logging.getLogger("serving.drift")
        self.reference = pd.read_parquet(resolve_path(cfg["data"]["reference_path"]))
        self.window: deque[dict[str, Any]] = deque(maxlen=cfg["drift"]["window_size"])
        self.check_every: int = cfg["drift"].get("check_every", 1)
        self.last_result: dict[str, Any] | None = None
        self._since_check = 0
        self._lock = threading.Lock()

    def add(self, row: dict[str, Any]) -> dict[str, Any] | None:
        """Append one raw input row; when the window is full, run :meth:`check`.

        Once full, the window is re-tested every ``drift.check_every`` requests:
        testing on every request would cost CPU and repeat the same WARNING.

        Returns the check result if one was run, else None.
        """
        with self._lock:
            self.window.append(row)
            self._since_check += 1
            due = len(self.window) == self.window.maxlen and self._since_check >= self.check_every
            if due:
                self._since_check = 0
        return self.check() if due else None

    def check(self) -> dict[str, Any]:
        """Run :func:`compare_frames` on the current window and store the result."""
        with self._lock:
            rows = list(self.window)
        window_size = self.window.maxlen
        if len(rows) < window_size:
            return {
                "status": "insufficient_data",
                "n_samples": len(rows),
                "window_size": window_size,
            }

        result = {"status": "ok", "window_size": window_size}
        result.update(compare_frames(self.reference, pd.DataFrame(rows), self.cfg))
        if result["drift_detected"]:
            self.logger.warning(
                "Data drift detected in %d features: %s",
                result["n_drifted"],
                ", ".join(result["drifted_features"]),
                extra={
                    "event": "drift_detected",
                    "drifted_features": result["drifted_features"],
                    "n_drifted": result["n_drifted"],
                },
            )
        else:
            self.logger.info("Drift check passed: %d features drifted", result["n_drifted"])
        self.last_result = result
        return result
