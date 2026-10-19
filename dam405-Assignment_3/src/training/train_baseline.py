"""Train the baseline Logistic Regression model and track it in MLflow.

The baseline is the reference point the AutoML model has to beat. It uses the
same preprocessor (``src.data.preprocess.build_preprocessor``) and the same
evaluation (``src.training.evaluate.evaluate_model``).

Run with ``python -m src.training.train_baseline``.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any

import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.config import load_config, mlflow_artifact_location, mlflow_tracking_uri, resolve_path
from src.data.preprocess import build_preprocessor, load_splits
from src.training.evaluate import evaluate_model
from src.utils.logging_config import setup_logging_from_config

logger = logging.getLogger("training.baseline")

RUN_NAME = "baseline_logreg"


def setup_mlflow(cfg: dict[str, Any]) -> str:
    """Point MLflow at the configured tracking URI and experiment; return experiment_id."""
    os.environ.setdefault("MLFLOW_ENABLE_ARTIFACTS_PROGRESS_BAR", "false")
    mlflow.set_tracking_uri(mlflow_tracking_uri(cfg))
    name = cfg["mlflow"]["experiment_name"]
    experiment = mlflow.get_experiment_by_name(name)
    if experiment is None:
        experiment_id = mlflow.create_experiment(
            name, artifact_location=mlflow_artifact_location(cfg)
        )
    else:
        experiment_id = experiment.experiment_id
    mlflow.set_experiment(experiment_id=experiment_id)
    return experiment_id


def log_sklearn_model(model: Any, X_train: Any) -> None:
    """Log a fitted pipeline as the run's ``model`` artifact with signature + input example."""
    mlflow.sklearn.log_model(
        model,
        artifact_path="model",
        signature=infer_signature(X_train, model.predict_proba(X_train)[:, 1]),
        # A few complete rows (NaN does not serialise to JSON).
        input_example=X_train.dropna().head(5),
        # Reuse our pins instead of letting MLflow infer them (slow, and less exact).
        pip_requirements=str(resolve_path("requirements.txt")),
    )


def train_baseline(cfg: dict[str, Any]) -> str:
    """Fit Pipeline(preprocessor, LogisticRegression), log everything, return the run_id."""
    X_train, X_val, X_test, y_train, y_val, y_test = load_splits(cfg)
    seed = cfg["project"]["random_seed"]
    bcfg = cfg["baseline"]
    figures_dir = resolve_path(cfg["explain"]["figures_dir"])

    pipeline = Pipeline(
        [
            ("preprocessor", build_preprocessor(cfg)),
            (
                "classifier",
                LogisticRegression(C=bcfg["C"], max_iter=bcfg["max_iter"], random_state=seed),
            ),
        ]
    )

    with mlflow.start_run(run_name=RUN_NAME) as run:
        mlflow.set_tags(
            {"model_type": "logistic_regression", "stage": "baseline", "dataset": "adult"}
        )
        mlflow.log_params(
            {
                "C": bcfg["C"],
                "max_iter": bcfg["max_iter"],
                "random_seed": seed,
                "n_train": len(X_train),
                "n_val": len(X_val),
                "n_test": len(X_test),
                "numeric_features": ",".join(cfg["data"]["numeric_features"]),
                "categorical_features": ",".join(cfg["data"]["categorical_features"]),
            }
        )

        start = time.perf_counter()
        pipeline.fit(X_train, y_train)
        fit_seconds = time.perf_counter() - start
        mlflow.log_metric("fit_time_seconds", fit_seconds)
        logger.info("Fitted baseline in %.2fs", fit_seconds)

        for prefix, X, y in (("val", X_val, y_val), ("test", X_test, y_test)):
            metrics, figures = evaluate_model(pipeline, X, y, prefix, figures_dir, RUN_NAME)
            mlflow.log_metrics(metrics)
            for fig in figures:
                mlflow.log_artifact(str(fig), artifact_path="plots")
            logger.info("%s metrics: %s", prefix, {k: round(v, 4) for k, v in metrics.items()})

        log_sklearn_model(pipeline, X_train)
        logger.info(
            "Baseline run %s finished: test_roc_auc=%.4f",
            run.info.run_id,
            mlflow.get_run(run.info.run_id).data.metrics["test_roc_auc"],
            extra={"run_id": run.info.run_id},
        )
        return run.info.run_id


def main() -> None:
    """Entry point."""
    global logger
    cfg = load_config()
    logger = setup_logging_from_config("training.baseline", cfg, "training_log")
    experiment_id = setup_mlflow(cfg)
    logger.info("MLflow experiment '%s' (id %s)", cfg["mlflow"]["experiment_name"], experiment_id)
    train_baseline(cfg)


if __name__ == "__main__":
    main()
