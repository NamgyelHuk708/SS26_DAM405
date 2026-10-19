"""API tests using FastAPI's TestClient.

A fake model is injected, so these tests need neither MLflow nor a registered
champion, and logs go to a temp directory (the real logs/service.log stays clean).
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from src.config import load_config
from src.serving.app import create_app
from src.serving.model_loader import ModelLoader
from src.serving.schemas import EXAMPLE_REQUEST


class FakeModel:
    """Stands in for the sklearn Pipeline: always predicts P(>50K) = 0.8."""

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return np.tile([0.2, 0.8], (len(X), 1))


@pytest.fixture
def valid_payload() -> dict[str, Any]:
    return dict(EXAMPLE_REQUEST)


@pytest.fixture
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    cfg = copy.deepcopy(load_config())
    cfg["logging"]["dir"] = str(tmp_path / "logs")
    ref_path = tmp_path / "reference.parquet"
    pd.DataFrame([EXAMPLE_REQUEST] * 20).to_parquet(ref_path, index=False)
    cfg["data"]["reference_path"] = str(ref_path)

    def fake_load(self: ModelLoader) -> FakeModel:
        self._model, self.model_version = FakeModel(), "test"
        return self._model

    monkeypatch.setattr(ModelLoader, "load", fake_load)
    with TestClient(create_app(cfg)) as c:
        yield c


def test_health(client: TestClient) -> None:
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True
    assert body["model_version"] == "test"


def test_predict_valid(client: TestClient, valid_payload: dict[str, Any]) -> None:
    r = client.post("/predict", json=valid_payload)
    assert r.status_code == 200
    body = r.json()
    assert body["prediction"] in (0, 1)
    assert 0 <= body["probability"] <= 1
    assert body["model_version"] == "test"
    assert body["request_id"] == r.headers["X-Request-ID"]


def test_predict_accepts_missing_category(client: TestClient, valid_payload: dict[str, Any]) -> None:
    r = client.post("/predict", json={**valid_payload, "workclass": None})
    assert r.status_code == 200


@pytest.mark.parametrize(
    "mutate",
    [
        lambda p: {k: v for k, v in p.items() if k != "age"},
        lambda p: {**p, "age": "abc"},
        lambda p: {**p, "hours-per-week": -5},
    ],
    ids=["missing_field", "wrong_type", "out_of_range"],
)
def test_predict_invalid(client: TestClient, valid_payload: dict[str, Any], mutate: Any) -> None:
    r = client.post("/predict", json=mutate(valid_payload))
    assert r.status_code == 422
    assert "request_id" in r.json()


def test_metrics_counts_requests_and_errors(
    client: TestClient, valid_payload: dict[str, Any]
) -> None:
    client.post("/predict", json=valid_payload)
    client.post("/predict", json={"age": "abc"})
    m = client.get("/metrics").json()
    # the /metrics request itself is recorded only after its response is built
    assert m["request_count"] >= 2
    assert m["error_count"] >= 1
    assert 0 < m["error_rate"] <= 1
    assert m["predictions_by_class"][">50K (1)"] == 1
    assert m["latency_ms"]["p95"] is not None


def test_drift_insufficient_data(client: TestClient) -> None:
    r = client.get("/drift")
    assert r.status_code == 200
    assert r.json()["status"] == "insufficient_data"
