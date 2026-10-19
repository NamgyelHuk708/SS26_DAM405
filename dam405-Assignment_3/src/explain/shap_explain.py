"""SHAP explanations (global + local) for the champion model.

Produces, in ``explain.figures_dir``:
* global: beeswarm summary, mean-|SHAP| bar chart, dependence plots for the
  top ``explain.top_k_features`` features;
* local: waterfall plots for one predicted-positive and one predicted-negative row.
All figures are also logged to MLflow under the champion's source run.

Run with ``python -m src.explain.shap_explain``.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import shap
from mlflow.tracking import MlflowClient
from sklearn.linear_model import LogisticRegression

from src.config import load_config, mlflow_tracking_uri, resolve_path
from src.data.preprocess import load_splits
from src.utils.logging_config import setup_logging_from_config

logger = logging.getLogger("explain.shap")


def load_champion(cfg: dict[str, Any]) -> tuple[Any, str]:
    """Return (sklearn Pipeline, source run_id) for the champion alias."""
    name, alias = cfg["mlflow"]["registered_model_name"], cfg["mlflow"]["champion_alias"]
    mv = MlflowClient().get_model_version_by_alias(name, alias)
    model = mlflow.sklearn.load_model(f"models:/{name}@{alias}")
    logger.info("Loaded %s@%s (version %s, run %s)", name, alias, mv.version, mv.run_id)
    return model, mv.run_id


def transform_sample(model: Any, X: pd.DataFrame, cfg: dict[str, Any]) -> pd.DataFrame:
    """Sample rows and transform them with the pipeline's preprocessor.

    Returns a DataFrame whose columns are the transformed feature names.
    """
    n = min(cfg["explain"]["sample_size"], len(X))
    X_sample = X.sample(n=n, random_state=cfg["project"]["random_seed"])
    pre = model.named_steps["preprocessor"]
    return pd.DataFrame(
        pre.transform(X_sample), columns=pre.get_feature_names_out(), index=X_sample.index
    )


def readable_feature_names(names: list[str], cfg: dict[str, Any]) -> list[str]:
    """Map one-hot names like ``cat__sex_Male`` to ``sex = Male`` and ``num__age`` to ``age``."""
    categorical = sorted(cfg["data"]["categorical_features"], key=len, reverse=True)
    readable = []
    for name in names:
        if name.startswith("num__"):
            readable.append(name[len("num__"):])
            continue
        rest = name[len("cat__"):] if name.startswith("cat__") else name
        feature = next((f for f in categorical if rest.startswith(f + "_")), None)
        readable.append(f"{feature} = {rest[len(feature) + 1:]}" if feature else rest)
    return readable


def compute_shap_values(model: Any, X_t: pd.DataFrame) -> shap.Explanation:
    """Return a ``shap.Explanation`` for the positive class."""
    est = model.named_steps["classifier"]
    if isinstance(est, LogisticRegression):
        explainer = shap.LinearExplainer(est, X_t)
    else:
        explainer = shap.TreeExplainer(est)
    explanation = explainer(X_t)
    if explanation.values.ndim == 3:  # one set of values per class -> keep class 1
        explanation = explanation[..., 1]
    return explanation


def _save(path: Path) -> Path:
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close("all")
    logger.info("Saved %s", path)
    return path


def save_global_plots(explanation: shap.Explanation, figures_dir: Path, top_k: int) -> list[Path]:
    """Save beeswarm, bar and top-k dependence plots; return their paths."""
    paths = []
    shap.plots.beeswarm(explanation, max_display=15, show=False)
    plt.title("SHAP summary (beeswarm): impact on P(income > 50K)")
    paths.append(_save(figures_dir / "shap_summary_beeswarm.png"))

    shap.plots.bar(explanation, max_display=15, show=False)
    plt.title("Mean |SHAP value|: global feature importance")
    paths.append(_save(figures_dir / "shap_bar.png"))

    mean_abs = np.abs(explanation.values).mean(axis=0)
    top = np.argsort(mean_abs)[::-1][:top_k]
    for rank, idx in enumerate(top, start=1):
        name = explanation.feature_names[idx]
        shap.plots.scatter(explanation[:, idx], color=explanation, show=False)
        plt.title(f"SHAP dependence #{rank}: {name}")
        safe = "".join(c if c.isalnum() else "_" for c in name).strip("_")
        paths.append(_save(figures_dir / f"shap_dependence_{rank}_{safe}.png"))
    return paths


def save_local_plots(
    model: Any, explanation: shap.Explanation, X_t: pd.DataFrame, figures_dir: Path
) -> list[Path]:
    """Save waterfall plots for one predicted-positive and one predicted-negative row."""
    proba = model.named_steps["classifier"].predict_proba(X_t.to_numpy())[:, 1]

    def typical(mask: np.ndarray) -> int:
        """Position of the row whose probability is the median within ``mask``.

        The most extreme rows (P ~ 0 or 1) are usually driven by one outlier such as a
        huge capital gain; a median case is more representative to interpret.
        """
        candidates = np.flatnonzero(mask)
        return int(candidates[np.argmin(np.abs(proba[candidates] - np.median(proba[candidates])))])

    cases = {"positive": typical(proba >= 0.5), "negative": typical(proba < 0.5)}
    paths = []
    for label, pos in cases.items():
        shap.plots.waterfall(explanation[pos], max_display=12, show=False)
        plt.title(
            f"Local explanation ({label}): P(>50K) = {proba[pos]:.3f}, test row {X_t.index[pos]}\n"
            "(SHAP values in log-odds; feature values shown unscaled)"
        )
        paths.append(_save(figures_dir / f"shap_waterfall_{label}.png"))
        logger.info("%s example: test row %s, P(>50K)=%.3f", label, X_t.index[pos], proba[pos])
    return paths


def main() -> None:
    """Entry point: explain the champion and log figures to its MLflow run."""
    global logger
    cfg = load_config()
    logger = setup_logging_from_config("explain.shap", cfg, "training_log")
    mlflow.set_tracking_uri(mlflow_tracking_uri(cfg))
    figures_dir = resolve_path(cfg["explain"]["figures_dir"])
    figures_dir.mkdir(parents=True, exist_ok=True)

    model, run_id = load_champion(cfg)
    _, _, X_test, _, _, _ = load_splits(cfg)
    X_t = transform_sample(model, X_test, cfg)
    X_t.columns = readable_feature_names(list(X_t.columns), cfg)
    logger.info("Explaining %d test rows x %d features", *X_t.shape)

    explanation = compute_shap_values(model, X_t)
    # The model sees standardised numbers (age 0.75); show readers the raw values (age 48).
    # This changes only the displayed feature values, not the SHAP values.
    display = X_t.copy()
    numeric = cfg["data"]["numeric_features"]
    display[numeric] = X_test.loc[X_t.index, numeric].to_numpy()
    explanation.data = display.to_numpy(dtype=float)

    importance = (
        pd.Series(np.abs(explanation.values).mean(axis=0), index=X_t.columns, name="mean_abs_shap")
        .sort_values(ascending=False)
    )
    importance_path = resolve_path("reports") / "shap_feature_importance.csv"
    importance.to_csv(importance_path, index_label="feature")
    logger.info("Top features: %s", importance.head(cfg["explain"]["top_k_features"]).round(4).to_dict())

    paths = save_global_plots(explanation, figures_dir, cfg["explain"]["top_k_features"])
    paths += save_local_plots(model, explanation, X_t, figures_dir)

    with mlflow.start_run(run_id=run_id):
        for p in paths:
            mlflow.log_artifact(str(p), artifact_path="shap")
        mlflow.log_artifact(str(importance_path), artifact_path="shap")
    logger.info("Logged %d SHAP figures to MLflow run %s", len(paths), run_id)


if __name__ == "__main__":
    main()
