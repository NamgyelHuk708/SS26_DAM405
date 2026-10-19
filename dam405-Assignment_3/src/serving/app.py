"""FastAPI prediction service for the champion Adult-income model.

Endpoints:
    GET  /health   liveness + whether the model is loaded and its version
    POST /predict  one prediction (PredictionRequest -> PredictionResponse)
    GET  /metrics  MetricsStore.snapshot()
    GET  /drift    DriftDetector result for the current window (or "insufficient_data")

Run with ``uvicorn src.serving.app:app --host 127.0.0.1 --port 8000`` (``make serve``).
"""

from __future__ import annotations

import time
import uuid
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator

from fastapi import FastAPI, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, RedirectResponse, Response

from src.config import load_config
from src.serving.drift import DriftDetector
from src.serving.model_loader import ModelLoader
from src.serving.monitoring import MetricsStore
from src.serving.schemas import PredictionRequest, PredictionResponse
from src.utils.logging_config import setup_logging_from_config

LABELS = {0: "<=50K", 1: ">50K"}


def create_app(cfg: dict[str, Any] | None = None) -> FastAPI:
    """Build and return the FastAPI app (a factory makes testing easier)."""
    cfg = cfg or load_config()
    logger = setup_logging_from_config("serving", cfg, "service_log")

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        logger.info("Service starting")
        app.state.model = ModelLoader(cfg, logger)
        app.state.model.load()  # logs CRITICAL and re-raises if it fails
        app.state.drift = DriftDetector(cfg, logger)
        logger.info(
            "Service ready: model version %s, drift window %d",
            app.state.model.model_version,
            app.state.drift.window.maxlen,
        )
        yield
        logger.info("Service shutting down")

    app = FastAPI(
        title="Adult Income Classifier",
        description="Predicts whether income > 50K. DAM405 PA3.",
        version="1.0.0",
        lifespan=lifespan,
    )
    app.state.metrics = MetricsStore()

    @app.middleware("http")
    async def observe(request: Request, call_next: Any) -> Any:
        request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
        request.state.request_id = request_id
        start = time.perf_counter()
        try:
            response = await call_next(request)
        except Exception as exc:  # unhandled error inside an endpoint
            logger.error(
                "Unhandled error: %s",
                exc,
                exc_info=exc,
                extra={"request_id": request_id, "endpoint": request.url.path},
            )
            response = JSONResponse(
                status_code=500,
                content={"detail": "Internal server error", "request_id": request_id},
            )
        latency_ms = (time.perf_counter() - start) * 1000
        status = response.status_code
        app.state.metrics.record_request(request.url.path, status, latency_ms)

        level = "error" if status >= 400 else "info"
        getattr(logger, level)(
            "%s %s -> %d (%.1f ms)",
            request.method,
            request.url.path,
            status,
            latency_ms,
            extra={
                "request_id": request_id,
                "method": request.method,
                "endpoint": request.url.path,
                "status_code": status,
                "latency_ms": round(latency_ms, 3),
            },
        )
        response.headers["X-Request-ID"] = request_id
        return response

    @app.exception_handler(RequestValidationError)
    async def validation_error(request: Request, exc: RequestValidationError) -> JSONResponse:
        errors = jsonable_encoder(exc.errors())
        logger.warning(
            "Invalid request: %d validation error(s)",
            len(errors),
            extra={
                "request_id": getattr(request.state, "request_id", None),
                "endpoint": request.url.path,
                "validation_errors": [
                    {"loc": e.get("loc"), "msg": e.get("msg")} for e in errors
                ],
            },
        )
        return JSONResponse(
            status_code=422,
            content={"detail": errors, "request_id": getattr(request.state, "request_id", None)},
        )

    @app.get("/", include_in_schema=False)
    def root() -> RedirectResponse:
        return RedirectResponse(url="/docs")

    @app.get("/favicon.ico", include_in_schema=False)
    def favicon() -> Response:
        # Browsers request this automatically; answer quietly instead of logging a 404 error.
        return Response(status_code=204)

    @app.get("/health")
    def health(request: Request) -> dict[str, Any]:
        model: ModelLoader = request.app.state.model
        return {
            "status": "ok" if model.is_loaded else "degraded",
            "model_loaded": model.is_loaded,
            "model_name": model.model_name,
            "model_version": model.model_version,
        }

    @app.post("/predict", response_model=PredictionResponse)
    def predict(payload: PredictionRequest, request: Request) -> PredictionResponse:
        state = request.app.state
        preds, probas = state.model.predict(payload.to_frame())
        prediction, probability = preds[0], probas[0]
        state.metrics.record_prediction(prediction)
        state.drift.add(payload.to_row())
        logger.debug(
            "Prediction %d (p=%.4f)",
            prediction,
            probability,
            extra={"request_id": request.state.request_id, "probability": probability},
        )
        return PredictionResponse(
            prediction=prediction,
            label=LABELS[prediction],
            probability=round(probability, 6),
            model_version=state.model.model_version,
            request_id=request.state.request_id,
        )

    @app.get("/metrics")
    def metrics(request: Request) -> dict[str, Any]:
        snapshot = request.app.state.metrics.snapshot()
        snapshot["model_version"] = request.app.state.model.model_version
        return snapshot

    @app.get("/drift")
    def drift(request: Request) -> dict[str, Any]:
        return request.app.state.drift.check()

    return app


app = create_app()
