"""Run FLAML AutoML on the preprocessed Adult data and track it in MLflow.

FLAML searches over the estimators in ``automl.estimator_list`` within
``automl.time_budget`` seconds. The winning estimator is wrapped together with
the already-fitted preprocessor into ONE sklearn ``Pipeline`` so the serving
layer can feed it raw feature rows, exactly like the baseline.

Run with ``python -m src.training.train_automl``.
"""

from __future__ import annotations

import logging
import time
from typing import Any

import mlflow
import pandas as pd
from flaml import AutoML
from flaml.automl.data import get_output_from_log
from sklearn.pipeline import Pipeline

from src.config import load_config, resolve_path
from src.data.preprocess import build_preprocessor, load_splits
from src.training.evaluate import evaluate_model
from src.training.train_baseline import log_sklearn_model, setup_mlflow
from src.utils.logging_config import setup_logging_from_config

logger = logging.getLogger("training.automl")

RUN_NAME = "automl_flaml"


def run_flaml(
    X_train_t: Any, y_train: pd.Series, X_val_t: Any, y_val: pd.Series, cfg: dict[str, Any]
) -> AutoML:
    """Fit and return a ``flaml.AutoML`` object on already-transformed features."""
    acfg = cfg["automl"]
    log_file = resolve_path(acfg["log_file"])
    log_file.parent.mkdir(parents=True, exist_ok=True)

    automl = AutoML()
    automl.fit(
        X_train=X_train_t,
        y_train=y_train.to_numpy(),
        X_val=X_val_t,
        y_val=y_val.to_numpy(),
        task=acfg["task"],
        metric=acfg["metric"],
        time_budget=acfg["time_budget"],
        estimator_list=acfg["estimator_list"],
        eval_method="holdout",
        log_file_name=str(log_file),
        seed=cfg["project"]["random_seed"],
        # We log a summary + the full search history ourselves; per-trial MLflow child
        # runs would add dozens of runs to the experiment.
        mlflow_logging=False,
        verbose=3,
    )
    return automl


def search_history(cfg: dict[str, Any]) -> pd.DataFrame:
    """Parse FLAML's log file into a DataFrame (one row per improving trial)."""
    acfg = cfg["automl"]
    times, best_losses, losses, configs, _metrics = get_output_from_log(
        filename=str(resolve_path(acfg["log_file"])), time_budget=acfg["time_budget"]
    )
    return pd.DataFrame(
        {
            "wall_clock_time": times,
            "valid_loss": losses,
            "best_valid_loss": best_losses,
            "learner": [c.get("Current Learner") for c in configs],
            "config": [c.get("Current Hyper-parameters") for c in configs],
        }
    )


def train_automl(cfg: dict[str, Any]) -> str:
    """Full AutoML training + MLflow logging; return the parent run_id."""
    setup_mlflow(cfg)
    X_train, X_val, X_test, y_train, y_val, y_test = load_splits(cfg)
    acfg = cfg["automl"]
    figures_dir = resolve_path(cfg["explain"]["figures_dir"])

    preprocessor = build_preprocessor(cfg).fit(X_train)
    X_train_t = preprocessor.transform(X_train)
    X_val_t = preprocessor.transform(X_val)
    logger.info("Transformed features: %d columns", X_train_t.shape[1])

    with mlflow.start_run(run_name=RUN_NAME) as run:
        mlflow.set_tags({"model_type": "flaml_automl", "stage": "automl", "dataset": "adult"})
        mlflow.log_params(
            {
                "time_budget": acfg["time_budget"],
                "metric": acfg["metric"],
                "estimator_list": ",".join(acfg["estimator_list"]),
                "random_seed": cfg["project"]["random_seed"],
                "n_train": len(X_train),
                "n_features_transformed": X_train_t.shape[1],
            }
        )

        logger.info(
            "Starting FLAML: budget=%ss, metric=%s, estimators=%s",
            acfg["time_budget"],
            acfg["metric"],
            acfg["estimator_list"],
        )
        start = time.perf_counter()
        automl = run_flaml(X_train_t, y_train, X_val_t, y_val, cfg)
        search_seconds = time.perf_counter() - start

        mlflow.set_tag("best_estimator", automl.best_estimator)
        mlflow.log_param("best_estimator", automl.best_estimator)
        mlflow.log_dict(automl.best_config, "best_config.json")
        mlflow.log_metrics(
            {
                "best_loss": float(automl.best_loss),
                f"best_val_{acfg['metric']}": 1 - float(automl.best_loss),
                "time_to_best_seconds": float(automl.time_to_find_best_model),
                "best_iteration": float(automl.best_iteration),
                "search_time_seconds": search_seconds,
            }
        )
        logger.info(
            "FLAML best: %s (val %s=%.4f) found after %.1fs; config=%s",
            automl.best_estimator,
            acfg["metric"],
            1 - automl.best_loss,
            automl.time_to_find_best_model,
            automl.best_config,
        )

        history = search_history(cfg)
        history_path = resolve_path("reports") / "flaml_search_history.csv"
        history.to_csv(history_path, index=False)
        mlflow.log_artifact(str(history_path))
        mlflow.log_artifact(str(resolve_path(acfg["log_file"])), artifact_path="flaml")

        final = Pipeline(
            [("preprocessor", preprocessor), ("classifier", automl.model.estimator)]
        )
        for prefix, X, y in (("val", X_val, y_val), ("test", X_test, y_test)):
            metrics, figures = evaluate_model(final, X, y, prefix, figures_dir, RUN_NAME)
            mlflow.log_metrics(metrics)
            for fig in figures:
                mlflow.log_artifact(str(fig), artifact_path="plots")
            logger.info("%s metrics: %s", prefix, {k: round(v, 4) for k, v in metrics.items()})

        log_sklearn_model(final, X_train)
        logger.info("AutoML run %s finished", run.info.run_id, extra={"run_id": run.info.run_id})
        return run.info.run_id


def main() -> None:
    """Entry point."""
    global logger
    cfg = load_config()
    logger = setup_logging_from_config("training.automl", cfg, "training_log")
    train_automl(cfg)


if __name__ == "__main__":
    main()
