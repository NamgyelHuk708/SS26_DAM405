"""Offline drift report: compare a batch CSV against the training reference.

Usage::

    python -m scripts.offline_drift_report --batch path/to/batch.csv

Writes a per-feature table to ``drift.report_path`` (reports/drift_report.csv)
and a reference-vs-batch histogram figure to ``explain.figures_dir``.
Reuses the same KS / PSI logic as the live service (``src.serving.drift``).
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

from src.config import load_config, resolve_path
from src.serving.drift import compare_frames


def parse_args() -> argparse.Namespace:
    """Parse ``--batch`` (required) and ``--output`` (optional, defaults to config)."""
    parser = argparse.ArgumentParser(description="Offline data-drift report vs. training reference.")
    parser.add_argument("--batch", required=True, help="CSV of raw feature rows to check")
    parser.add_argument("--output", default=None, help="Output CSV (default: drift.report_path)")
    return parser.parse_args()


def build_report(reference: pd.DataFrame, batch: pd.DataFrame, cfg: dict[str, Any]) -> pd.DataFrame:
    """Return one row per feature: feature, type, test, statistic, p_value, threshold, drifted."""
    result = compare_frames(reference, batch, cfg)
    report = pd.DataFrame(
        [{"feature": name, **values} for name, values in result["features"].items()]
    )
    report = report[["feature", "type", "test", "statistic", "p_value", "threshold", "drifted"]]
    return report.sort_values(["drifted", "type"], ascending=[False, True]).reset_index(drop=True)


def save_histograms(
    reference: pd.DataFrame, batch: pd.DataFrame, features: list[str], out_path: Path
) -> Path:
    """Overlayed reference-vs-batch histograms for the given numeric features."""
    fig, axes = plt.subplots(1, len(features), figsize=(5 * len(features), 4), squeeze=False)
    for ax, col in zip(axes[0], features):
        ax.hist(reference[col].dropna(), bins=30, density=True, alpha=0.5, label="reference (train)")
        ax.hist(batch[col].dropna(), bins=30, density=True, alpha=0.5, label="batch (live)")
        ax.set_title(col)
        ax.legend()
    fig.suptitle("Reference vs. live distributions")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main() -> None:
    """Load reference + batch, build the report, save it, print a summary."""
    args = parse_args()
    cfg = load_config()
    numeric = cfg["data"]["numeric_features"]
    features = numeric + cfg["data"]["categorical_features"]

    reference = pd.read_parquet(resolve_path(cfg["data"]["reference_path"]))
    batch = pd.read_csv(resolve_path(args.batch))[features]

    report = build_report(reference, batch, cfg)
    out = Path(args.output) if args.output else resolve_path(cfg["drift"]["report_path"])
    out.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(out, index=False)

    figures_dir = resolve_path(cfg["explain"]["figures_dir"])
    figures_dir.mkdir(parents=True, exist_ok=True)
    plot_features = ["age", "hours-per-week", "capital-gain"]
    plot_features = [f for f in plot_features if f in numeric] or numeric[:3]
    fig_path = save_histograms(reference, batch, plot_features, figures_dir / "drift_histograms.png")

    n_drifted = int(report["drifted"].sum())
    overall = n_drifted >= cfg["drift"]["min_drifted_features"]
    with pd.option_context("display.width", 120):
        print(report.to_string(index=False))
    print(f"\nBatch: {args.batch} ({len(batch)} rows) vs reference ({len(reference)} rows)")
    print(f"Drifted features: {n_drifted}/{len(report)} -> overall drift: {overall}")
    print(f"Saved report to {out}")
    print(f"Saved histograms to {fig_path}")


if __name__ == "__main__":
    main()
