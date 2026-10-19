"""Download, clean and cache the UCI Adult Income dataset.

Run with ``python -m src.data.load_data``. The first run fetches the data from
OpenML and caches the cleaned frame at ``data.raw_path``; later runs read the
cache. The target column is mapped to 0/1 (1 = income >50K).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml

from src.config import load_config, resolve_path
from src.utils.logging_config import setup_logging_from_config


def _clean(df: pd.DataFrame, target: str, positive_label: str) -> pd.DataFrame:
    df = df.copy()
    for col in df.columns:
        if isinstance(df[col].dtype, pd.CategoricalDtype) or df[col].dtype == object:
            df[col] = df[col].astype("string").str.strip().astype(object)
    df = df.replace("?", np.nan)
    # Some Adult variants write labels as ">50K." — normalise the trailing dot.
    labels = df[target].astype(str).str.rstrip(".")
    df[target] = (labels == positive_label).astype(int)
    return df


def load_adult(force_download: bool = False) -> pd.DataFrame:
    """Return the cleaned Adult dataframe (features + 0/1 target)."""
    cfg = load_config()
    logger = setup_logging_from_config("data.load", cfg, "training_log")
    dcfg = cfg["data"]
    raw_path = resolve_path(dcfg["raw_path"])
    target = dcfg["target"]

    if raw_path.exists() and not force_download:
        logger.info("Loading cached dataset from %s", raw_path)
        df = pd.read_csv(raw_path)
    else:
        logger.info(
            "Fetching OpenML dataset '%s' (version %s)", dcfg["openml_name"], dcfg["openml_version"]
        )
        bunch = fetch_openml(
            dcfg["openml_name"], version=dcfg["openml_version"], as_frame=True, parser="auto"
        )
        df = bunch.frame.rename(columns={bunch.target.name: target})
        df = _clean(df, target, dcfg["positive_label"])
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(raw_path, index=False)
        logger.info("Cached cleaned dataset to %s", raw_path)

    expected = set(dcfg["numeric_features"]) | set(dcfg["categorical_features"]) | {target}
    missing = expected - set(df.columns)
    if missing:
        raise KeyError(f"Columns from config not found in dataset: {sorted(missing)}")

    balance = df[target].value_counts(normalize=True).sort_index()
    logger.info(
        "Dataset shape: %s rows x %s cols",
        df.shape[0],
        df.shape[1],
        extra={"rows": df.shape[0], "cols": df.shape[1]},
    )
    logger.info(
        "Class balance: <=50K (0) = %.3f, >50K (1) = %.3f",
        balance.get(0, 0.0),
        balance.get(1, 0.0),
        extra={"class_balance": {int(k): round(float(v), 4) for k, v in balance.items()}},
    )
    logger.info(
        "Missing values per column: %s",
        {c: int(n) for c, n in df.isna().sum().items() if n > 0},
    )
    return df


if __name__ == "__main__":
    load_adult()
