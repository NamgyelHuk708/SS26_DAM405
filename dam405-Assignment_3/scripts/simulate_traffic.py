"""Send synthetic traffic to the running API to exercise monitoring and drift.

Three phases:
1. ``simulation.n_normal`` normal requests sampled from the test split.
2. ``simulation.n_drifted`` drifted requests: numeric features shifted and the
   category mix changed (should make GET /drift report drift).
3. ``simulation.n_malformed`` malformed requests (missing fields, wrong types)
   to produce 4xx errors and raise the error rate.

The drifted batch is also saved to ``data/processed/drifted_batch.csv`` so the
offline drift report (``make drift-report``) can analyse the same data.

Start the API first (``make serve``), then run ``make simulate``.
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from typing import Any

import numpy as np
import pandas as pd
import requests

from src.config import load_config, resolve_path
from src.data.preprocess import load_splits


def make_drifted(X: pd.DataFrame, cfg: dict[str, Any], seed: int) -> pd.DataFrame:
    """Return a copy of X with a deliberate distribution shift.

    Shifts (documented in the README):
    * age + 15 years (capped at 90)
    * hours-per-week x 1.5 (capped at 99)
    * capital-gain: 40% of rows get a random gain of 5,000–20,000
    * workclass: 70% of rows set to "Self-emp-inc"
    * marital-status: re-sampled, 70% "Never-married"
    """
    rng = np.random.default_rng(seed)
    X = X.copy()
    n = len(X)
    X["age"] = (X["age"] + 15).clip(upper=90)
    X["hours-per-week"] = (X["hours-per-week"] * 1.5).round().clip(upper=99).astype(int)
    gain_mask = rng.random(n) < 0.4
    X.loc[gain_mask, "capital-gain"] = rng.integers(5_000, 20_000, gain_mask.sum())
    X.loc[rng.random(n) < 0.7, "workclass"] = "Self-emp-inc"
    X["marital-status"] = rng.choice(
        ["Never-married", "Married-civ-spouse", "Divorced", "Separated"],
        size=n,
        p=[0.7, 0.1, 0.1, 0.1],
    )
    return X


def to_payload(row: pd.Series) -> dict[str, Any]:
    """Convert one feature row to a JSON-safe dict (NaN -> None, numpy -> Python)."""
    out: dict[str, Any] = {}
    for key, value in row.items():
        if value is None or (isinstance(value, float) and np.isnan(value)):
            out[key] = None
        elif isinstance(value, (np.integer, np.floating)):
            out[key] = int(value)
        else:
            out[key] = value
    return out


def malformed_payloads(valid_example: dict[str, Any], n: int) -> list[dict[str, Any]]:
    """Return n invalid payloads (missing field, wrong type, out-of-range value...)."""
    variants = [
        {k: v for k, v in valid_example.items() if k != "age"},  # missing field
        {**valid_example, "age": "abc"},  # wrong type
        {**valid_example, "hours-per-week": -5},  # out of range
        {**valid_example, "education-num": 42},  # out of range
        {**valid_example, "unexpected_field": 1},  # unknown field
        {},  # empty body
    ]
    return [variants[i % len(variants)] for i in range(n)]


def send(session: requests.Session, base_url: str, rows: list[dict[str, Any]], label: str) -> Counter:
    """POST each row to /predict; return counts of status codes."""
    counts: Counter = Counter()
    for i, row in enumerate(rows, start=1):
        try:
            status = session.post(f"{base_url}/predict", json=row, timeout=10).status_code
        except requests.RequestException:
            status = "connection_error"
        counts[status] += 1
        if i % 100 == 0 or i == len(rows):
            print(f"  [{label}] sent {i}/{len(rows)}  status counts: {dict(counts)}")
    return counts


def show_drift(session: requests.Session, base_url: str, title: str) -> dict[str, Any]:
    """Print a compact summary of GET /drift."""
    drift = session.get(f"{base_url}/drift", timeout=30).json()
    print(f"\n--- GET /drift after {title} ---")
    if drift.get("status") != "ok":
        print(json.dumps(drift, indent=2))
        return drift
    print(f"drift_detected={drift['drift_detected']}  n_drifted={drift['n_drifted']}  "
          f"drifted_features={drift['drifted_features']}")
    return drift


def main() -> None:
    """Run the three phases and print a summary (incl. GET /metrics and GET /drift)."""
    cfg = load_config()
    scfg = cfg["simulation"]
    seed = cfg["project"]["random_seed"]
    base_url = f"http://{cfg['serving']['host']}:{cfg['serving']['port']}"
    session = requests.Session()

    try:
        health = session.get(f"{base_url}/health", timeout=5).json()
    except (requests.RequestException, ValueError):
        sys.exit(f"API not reachable at {base_url}. Start it first with `make serve`.")
    if "model_loaded" not in health:
        sys.exit(f"{base_url} is answering, but it is not this project's API: {health}")
    print(f"API healthy: {health}")

    _, _, X_test, _, _, _ = load_splits(cfg)
    normal = X_test.sample(n=scfg["n_normal"], random_state=seed)
    drifted = make_drifted(X_test.sample(n=scfg["n_drifted"], random_state=seed + 1), cfg, seed)
    drifted_path = resolve_path(cfg["data"]["processed_dir"]) / "drifted_batch.csv"
    drifted.to_csv(drifted_path, index=False)

    results = {}
    print(f"\nPhase 1: {len(normal)} normal requests")
    results["normal"] = send(session, base_url, [to_payload(r) for _, r in normal.iterrows()], "normal")
    show_drift(session, base_url, "NORMAL traffic")

    print(f"\nPhase 2: {len(drifted)} drifted requests (saved to {drifted_path})")
    results["drifted"] = send(session, base_url, [to_payload(r) for _, r in drifted.iterrows()], "drifted")
    show_drift(session, base_url, "DRIFTED traffic")

    print(f"\nPhase 3: {scfg['n_malformed']} malformed requests")
    example = to_payload(normal.iloc[0])
    results["malformed"] = send(session, base_url, malformed_payloads(example, scfg["n_malformed"]), "malformed")

    print("\n=== Summary: status codes per phase ===")
    for phase, counts in results.items():
        print(f"  {phase:<10} {dict(counts)}")
    print("\n--- GET /metrics ---")
    print(json.dumps(session.get(f"{base_url}/metrics", timeout=5).json(), indent=2))


if __name__ == "__main__":
    main()
