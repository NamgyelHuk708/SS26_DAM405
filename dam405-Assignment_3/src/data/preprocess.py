from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from src.config import load_config, resolve_path
from src.data.load_data import load_adult
from src.utils.logging_config import setup_logging_from_config

logger = logging.getLogger("data.preprocess")

SPLITS = ("train", "val", "test")


def feature_columns(cfg: dict[str, Any]) -> list[str]:
    """Numeric + categorical feature names, in config order."""
    return list(cfg["data"]["numeric_features"]) + list(cfg["data"]["categorical_features"])


def none_to_nan(X: Any) -> pd.DataFrame:
    """Replace ``None`` with ``np.nan``.

    Parquet files and JSON requests represent a missing category as ``None``,
    which ``SimpleImputer`` does not treat as missing. Running this first makes
    training and serving handle missing values identically. It must stay a
    module-level function so the fitted pipeline can be pickled by MLflow.
    """
    df = pd.DataFrame(X).astype(object)
    return df.where(df.notna(), np.nan)


def split_data(
    df: pd.DataFrame, cfg: dict[str, Any]
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """Stratified split into (X_train, X_val, X_test, y_train, y_val, y_test).

    ``test_size`` and ``val_size`` are fractions of the FULL dataset.
    """
    d = cfg["data"]
    seed = cfg["project"]["random_seed"]
    X = df[feature_columns(cfg)]
    y = df[d["target"]]

    X_rest, X_test, y_rest, y_test = train_test_split(
        X, y, test_size=d["test_size"], stratify=y, random_state=seed
    )
    # val_size is relative to the full data, but this split only sees the remainder.
    val_fraction = d["val_size"] / (1 - d["test_size"])
    X_train, X_val, y_train, y_val = train_test_split(
        X_rest, y_rest, test_size=val_fraction, stratify=y_rest, random_state=seed
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


def build_preprocessor(cfg: dict[str, Any]) -> ColumnTransformer:
    """Return an UNFITTED ColumnTransformer for the Adult features."""
    numeric = Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]
    )
    categorical = Pipeline(
        [
            ("none_to_nan", FunctionTransformer(none_to_nan, feature_names_out="one-to-one")),
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    return ColumnTransformer(
        [
            ("num", numeric, list(cfg["data"]["numeric_features"])),
            ("cat", categorical, list(cfg["data"]["categorical_features"])),
        ]
    )


def save_splits(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_val: pd.Series,
    y_test: pd.Series,
    cfg: dict[str, Any],
) -> None:
    """Write the splits to ``data.processed_dir`` and the reference snapshot."""
    target = cfg["data"]["target"]
    processed_dir = resolve_path(cfg["data"]["processed_dir"])
    processed_dir.mkdir(parents=True, exist_ok=True)

    for name, X, y in zip(SPLITS, (X_train, X_val, X_test), (y_train, y_val, y_test)):
        path = processed_dir / f"{name}.parquet"
        X.assign(**{target: y}).to_parquet(path, index=False)
        logger.info(
            "Saved %-5s split: %6d rows, positive rate %.4f -> %s",
            name,
            len(X),
            y.mean(),
            path,
            extra={"split": name, "rows": len(X), "positive_rate": round(float(y.mean()), 4)},
        )

    reference_path = resolve_path(cfg["data"]["reference_path"])
    reference_path.parent.mkdir(parents=True, exist_ok=True)
    X_train.to_parquet(reference_path, index=False)
    logger.info("Saved drift reference (raw training features) -> %s", reference_path)


def load_splits(
    cfg: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """Read the parquet splits written by :func:`save_splits`."""
    target = cfg["data"]["target"]
    processed_dir = resolve_path(cfg["data"]["processed_dir"])
    Xs, ys = [], []
    for name in SPLITS:
        path = processed_dir / f"{name}.parquet"
        if not path.exists():
            raise FileNotFoundError(f"{path} not found. Run `make data` first.")
        df = pd.read_parquet(path)
        Xs.append(df.drop(columns=target))
        ys.append(df[target])
    return Xs[0], Xs[1], Xs[2], ys[0], ys[1], ys[2]


def main() -> None:
    """Entry point: load raw data, split, save."""
    global logger
    cfg = load_config()
    logger = setup_logging_from_config("data.preprocess", cfg, "training_log")

    df = load_adult()
    splits = split_data(df, cfg)
    save_splits(*splits, cfg)

    X_train = splits[0]
    n_out = build_preprocessor(cfg).fit(X_train).get_feature_names_out().size
    logger.info(
        "Preprocessor check: %d raw features -> %d model features after one-hot encoding",
        X_train.shape[1],
        n_out,
    )


if __name__ == "__main__":
    main()
