"""Select the best tracked run and register it in the MLflow Model Registry.

The run with the highest ``mlflow.selection_metric`` (test ROC-AUC) is
registered as ``mlflow.registered_model_name`` and receives the alias
``mlflow.champion_alias``. The serving layer loads
``models:/<name>@<alias>``, so promoting a new model needs no code change.

Run with ``python -m src.training.register_model``.
"""

from __future__ import annotations

import logging
from datetime import date
from typing import Any

import mlflow
from mlflow.tracking import MlflowClient

from src.config import load_config, mlflow_tracking_uri
from src.utils.logging_config import setup_logging_from_config

logger = logging.getLogger("training.register")


def find_best_run(cfg: dict[str, Any]) -> Any:
    """Return the best ``mlflow.entities.Run`` in the experiment."""
    client = MlflowClient(tracking_uri=mlflow_tracking_uri(cfg))
    metric = cfg["mlflow"]["selection_metric"]
    experiment = client.get_experiment_by_name(cfg["mlflow"]["experiment_name"])
    if experiment is None:
        raise RuntimeError("Experiment not found. Run `make baseline` and `make automl` first.")

    runs = client.search_runs(
        [experiment.experiment_id],
        filter_string=f"metrics.{metric} > 0",
        order_by=[f"metrics.{metric} DESC"],
        max_results=50,
    )
    if not runs:
        raise RuntimeError(f"No runs with metric '{metric}'. Run `make baseline` / `make automl`.")

    logger.info("Leaderboard by %s:", metric)
    for rank, r in enumerate(runs, start=1):
        logger.info(
            "  %d. %-16s %s=%.4f  (run %s)",
            rank,
            r.info.run_name,
            metric,
            r.data.metrics[metric],
            r.info.run_id,
        )
    return runs[0]


def register_champion(cfg: dict[str, Any]) -> Any:
    """Register the best run's model, set the champion alias, return the ModelVersion."""
    mcfg = cfg["mlflow"]
    name, alias, metric = mcfg["registered_model_name"], mcfg["champion_alias"], mcfg["selection_metric"]
    mlflow.set_tracking_uri(mlflow_tracking_uri(cfg))
    client = MlflowClient()

    best = find_best_run(cfg)
    score = best.data.metrics[metric]
    model_type = best.data.tags.get("model_type", "unknown")

    mv = mlflow.register_model(f"runs:/{best.info.run_id}/model", name=name)
    client.set_registered_model_alias(name, alias, mv.version)
    client.update_registered_model(
        name, description="Binary classifier for UCI Adult income (>50K = 1). DAM405 PA3."
    )
    client.update_model_version(
        name,
        mv.version,
        description=(
            f"{best.info.run_name} ({model_type}) selected on {date.today().isoformat()} "
            f"as best run by {metric} = {score:.4f}."
        ),
    )
    for key, value in {
        "model_type": model_type,
        metric: f"{score:.4f}",
        "source_run_name": best.info.run_name,
        "best_estimator": best.data.tags.get("best_estimator", model_type),
    }.items():
        client.set_model_version_tag(name, mv.version, key, value)

    logger.info(
        "Registered %s version %s from run '%s' (%s=%.4f) with alias @%s",
        name,
        mv.version,
        best.info.run_name,
        metric,
        score,
        alias,
        extra={"model_name": name, "model_version": mv.version, "alias": alias},
    )
    return mv


def main() -> None:
    """Entry point."""
    global logger
    cfg = load_config()
    logger = setup_logging_from_config("training.register", cfg, "training_log")
    register_champion(cfg)


if __name__ == "__main__":
    main()
