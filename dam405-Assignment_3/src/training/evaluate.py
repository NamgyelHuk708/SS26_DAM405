"""Shared evaluation helpers for every model in the project.

Both the baseline and the AutoML model are scored with :func:`evaluate_model`,
so their MLflow metrics are directly comparable.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")  # no display needed; figures are only saved to disk

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    accuracy_score,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)

CLASS_LABELS = ["<=50K", ">50K"]


def compute_metrics(y_true: pd.Series, y_pred: Any, y_proba: Any) -> dict[str, float]:
    """Return accuracy, precision, recall, f1, roc_auc and log_loss.

    Args:
        y_true: True 0/1 labels.
        y_pred: Predicted 0/1 labels.
        y_proba: Predicted probability of class 1.
    """
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred)),
        "roc_auc": float(roc_auc_score(y_true, y_proba)),
        "log_loss": float(log_loss(y_true, y_proba)),
    }


def save_confusion_matrix(y_true: pd.Series, y_pred: Any, out_path: Path, title: str) -> Path:
    """Save a confusion-matrix PNG and return its path."""
    disp = ConfusionMatrixDisplay.from_predictions(
        y_true, y_pred, display_labels=CLASS_LABELS, cmap="Blues", values_format="d"
    )
    disp.ax_.set_title(title)
    disp.figure_.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(disp.figure_)
    return out_path


def save_roc_curve(y_true: pd.Series, y_proba: Any, out_path: Path, title: str) -> Path:
    """Save a ROC-curve PNG and return its path."""
    disp = RocCurveDisplay.from_predictions(y_true, y_proba)
    disp.ax_.plot([0, 1], [0, 1], linestyle="--", color="grey", label="Chance")
    disp.ax_.set_title(title)
    disp.ax_.legend(loc="lower right")
    disp.figure_.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(disp.figure_)
    return out_path


def evaluate_model(
    model: Any,
    X: pd.DataFrame,
    y: pd.Series,
    prefix: str,
    figures_dir: Path,
    model_name: str,
) -> tuple[dict[str, float], list[Path]]:
    """Evaluate a fitted pipeline on (X, y).

    Returns:
        (metrics, figure_paths) where metric keys are prefixed, e.g. ``test_roc_auc``.
        The prefix matters: register_model.py selects the champion by ``test_roc_auc``.
    """
    y_proba = model.predict_proba(X)[:, 1]
    y_pred = model.predict(X)
    metrics = {f"{prefix}_{k}": v for k, v in compute_metrics(y, y_pred, y_proba).items()}

    figures_dir.mkdir(parents=True, exist_ok=True)
    cm_path = save_confusion_matrix(
        y,
        y_pred,
        figures_dir / f"{model_name}_{prefix}_confusion_matrix.png",
        f"{model_name}: confusion matrix ({prefix})",
    )
    roc_path = save_roc_curve(
        y,
        y_proba,
        figures_dir / f"{model_name}_{prefix}_roc.png",
        f"{model_name}: ROC curve ({prefix})",
    )
    return metrics, [cm_path, roc_path]
