"""Load and cache the champion model from the MLflow Model Registry.

The service never points at a run id or a file path: it loads
``models:/<registered_model_name>@<champion_alias>``, so re-pointing the alias
(``register_model.py``) is all it takes to deploy a new model.
"""

from __future__ import annotations

import logging
import threading
from typing import Any

import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient

from src.config import mlflow_tracking_uri


class ModelLoader:
    """Loads the champion once and exposes it plus its registry version."""

    def __init__(self, cfg: dict[str, Any], logger: logging.Logger | None = None) -> None:
        """Store config; do NOT load the model here (do it in :meth:`load`)."""
        self.cfg = cfg
        self.logger = logger or logging.getLogger("serving.model")
        self.model_name: str = cfg["mlflow"]["registered_model_name"]
        self.alias: str = cfg["mlflow"]["champion_alias"]
        self.model_uri = f"models:/{self.model_name}@{self.alias}"
        self.model_version: str | None = None
        self._model: Any = None
        self._lock = threading.Lock()

    def load(self) -> Any:
        """Load (or return the cached) sklearn Pipeline."""
        if self._model is not None:
            return self._model
        with self._lock:
            if self._model is None:
                try:
                    mlflow.set_tracking_uri(mlflow_tracking_uri(self.cfg))
                    model = mlflow.sklearn.load_model(self.model_uri)
                    mv = MlflowClient().get_model_version_by_alias(self.model_name, self.alias)
                except Exception:
                    self.logger.critical(
                        "Could not load model %s; the service cannot serve predictions. "
                        "Did you run `make register`?",
                        self.model_uri,
                        exc_info=True,
                    )
                    raise
                self._model, self.model_version = model, str(mv.version)
                self.logger.info(
                    "Loaded model %s (version %s)",
                    self.model_uri,
                    self.model_version,
                    extra={"model_uri": self.model_uri, "model_version": self.model_version},
                )
        return self._model

    @property
    def is_loaded(self) -> bool:
        """True once :meth:`load` succeeded (used by GET /health)."""
        return self._model is not None

    def predict(self, X: Any) -> tuple[list[int], list[float]]:
        """Return (predicted classes, probability of class 1) for a DataFrame of raw rows."""
        proba = self.load().predict_proba(X)[:, 1]
        preds = (proba >= 0.5).astype(int)
        return preds.tolist(), proba.tolist()
