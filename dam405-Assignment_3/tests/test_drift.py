"""Tests for src/serving/drift.py: no drift on same distribution, drift on shifted data."""

from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.config import load_config
from src.serving.drift import DriftDetector, compare_frames, psi

CATEGORIES = ["a", "b", "c", "d"]
BASE_PROBS = [0.4, 0.3, 0.2, 0.1]


def make_frame(cfg: dict, n: int, seed: int, shift: float = 0.0, probs=None) -> pd.DataFrame:
    """Synthetic data with the configured columns; ``shift`` moves numeric means (in SDs)."""
    rng = np.random.default_rng(seed)
    data = {col: rng.normal(50 + shift * 10, 10, n) for col in cfg["data"]["numeric_features"]}
    for col in cfg["data"]["categorical_features"]:
        data[col] = rng.choice(CATEGORIES, size=n, p=probs or BASE_PROBS)
    return pd.DataFrame(data)


@pytest.fixture
def cfg() -> dict:
    return load_config()


@pytest.fixture
def reference_frame(cfg: dict) -> pd.DataFrame:
    return make_frame(cfg, n=2000, seed=cfg["project"]["random_seed"])


def test_same_distribution_no_drift(cfg: dict, reference_frame: pd.DataFrame) -> None:
    current = make_frame(cfg, n=cfg["drift"]["window_size"], seed=123)
    result = compare_frames(reference_frame, current, cfg)
    assert result["drift_detected"] is False
    assert result["n_drifted"] < cfg["drift"]["min_drifted_features"]


def test_shifted_distribution_detects_drift(cfg: dict, reference_frame: pd.DataFrame) -> None:
    current = make_frame(
        cfg, n=cfg["drift"]["window_size"], seed=456, shift=2.0, probs=[0.05, 0.05, 0.1, 0.8]
    )
    result = compare_frames(reference_frame, current, cfg)
    assert result["drift_detected"] is True
    assert result["n_drifted"] >= cfg["drift"]["min_drifted_features"]
    for col in cfg["data"]["numeric_features"]:
        assert result["features"][col]["drifted"], col
    for col in cfg["data"]["categorical_features"]:
        assert result["features"][col]["statistic"] > cfg["drift"]["psi_threshold"], col


def test_psi_is_zero_for_identical_series() -> None:
    s = pd.Series(["a", "b", "b", "c", None])
    assert psi(s, s) == pytest.approx(0, abs=1e-9)


def test_psi_handles_unseen_category() -> None:
    ref = pd.Series(["a"] * 50 + ["b"] * 50)
    cur = pd.Series(["c"] * 100)  # category never seen in the reference
    assert psi(ref, cur) > 1.0


def test_psi_pools_rare_categories() -> None:
    rng = np.random.default_rng(0)
    common, rare = ["a", "b"], [f"r{i}" for i in range(40)]  # 40 categories at 0.5% each
    probs = [0.4, 0.4] + [0.2 / 40] * 40
    ref = pd.Series(rng.choice(common + rare, size=20_000, p=probs))
    cur = pd.Series(rng.choice(common + rare, size=200, p=probs))  # same distribution
    assert psi(ref, cur, min_share=0.01) < psi(ref, cur)
    assert psi(ref, cur, min_share=0.01) < 0.1


def test_detector_insufficient_data_then_ok(cfg: dict, tmp_path: Path) -> None:
    cfg = copy.deepcopy(cfg)
    cfg["drift"]["window_size"] = 50
    cfg["drift"]["check_every"] = 1
    ref_path = tmp_path / "reference.parquet"
    make_frame(cfg, n=500, seed=1).to_parquet(ref_path, index=False)
    cfg["data"]["reference_path"] = str(ref_path)

    detector = DriftDetector(cfg)
    rows = make_frame(cfg, n=50, seed=2).to_dict(orient="records")
    for row in rows[:10]:
        detector.add(row)
    assert detector.check()["status"] == "insufficient_data"

    for row in rows[10:]:
        detector.add(row)
    result = detector.check()
    assert result["status"] == "ok"
    assert result["n_samples"] == 50
