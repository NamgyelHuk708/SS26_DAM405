"""Structured logging setup shared by training scripts and the API service.

Two handlers are attached to each named logger:

* **Console** (level INFO) — human-readable:
  ``%(asctime)s | %(levelname)-8s | %(name)s | %(message)s``
* **Rotating file** (level DEBUG) — one JSON object per line (JSON Lines), so
  logs can be parsed with pandas / jq for monitoring evidence.

Log-level conventions used throughout the project
-------------------------------------------------
DEBUG     Internal detail useful only when debugging (shapes, intermediate values).
INFO      Normal events: a request served, a training run finished, a model loaded.
WARNING   Degraded but still working: data drift detected, slow request, fallback used.
ERROR     A single operation failed: invalid/failed request, exception in an endpoint.
CRITICAL  The service cannot serve at all, e.g. the model failed to load at startup.

Usage::

    from src.utils.logging_config import setup_logging
    logger = setup_logging("serving", log_file="logs/service.log")
    logger.info("request served", extra={"request_id": rid, "latency_ms": 12.3})
"""

from __future__ import annotations

import json
import logging
import logging.config
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.config import resolve_path

# Attributes present on every LogRecord; anything else came from ``extra=``.
_RESERVED_ATTRS: frozenset[str] = frozenset(
    vars(logging.LogRecord("", 0, "", 0, "", None, None)).keys()
) | {"message", "asctime", "taskName"}

CONSOLE_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"


class JsonFormatter(logging.Formatter):
    """Format a LogRecord as a single-line JSON object."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
            "message": record.getMessage(),
        }
        for key, value in record.__dict__.items():
            if key not in _RESERVED_ATTRS and not key.startswith("_"):
                payload[key] = value
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        if record.stack_info:
            payload["stack_info"] = self.formatStack(record.stack_info)
        return json.dumps(payload, default=str, ensure_ascii=False)


def setup_logging(
    name: str,
    log_file: str | Path | None = None,
    level: str = "INFO",
    max_bytes: int = 5 * 1024 * 1024,
    backup_count: int = 3,
) -> logging.Logger:
    """Configure and return the logger ``name``.

    Args:
        name: Logger name (e.g. ``"serving"``, ``"training.baseline"``).
        log_file: Path of the JSON-lines log file, relative to the project root.
            If ``None``, only the console handler is attached.
        level: Level of the logger itself (handlers filter further:
            console = INFO, file = DEBUG).
        max_bytes: Rotate the file after this many bytes.
        backup_count: Number of rotated files to keep.

    Calling this twice for the same logger replaces its handlers instead of
    duplicating them.
    """
    handlers: dict[str, dict[str, Any]] = {
        "console": {
            "class": "logging.StreamHandler",
            "level": "INFO",
            "formatter": "human",
            "stream": "ext://sys.stderr",
        }
    }
    if log_file is not None:
        path = resolve_path(log_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        handlers["file"] = {
            "class": "logging.handlers.RotatingFileHandler",
            "level": "DEBUG",
            "formatter": "json",
            "filename": str(path),
            "maxBytes": max_bytes,
            "backupCount": backup_count,
            "encoding": "utf-8",
        }

    # Close existing handlers so files aren't left open / duplicated.
    existing = logging.getLogger(name)
    for h in list(existing.handlers):
        existing.removeHandler(h)
        h.close()

    logging.config.dictConfig(
        {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "human": {"format": CONSOLE_FORMAT},
                "json": {"()": JsonFormatter},
            },
            "handlers": handlers,
            "loggers": {
                name: {
                    "level": level.upper(),
                    "handlers": list(handlers),
                    "propagate": False,
                }
            },
        }
    )
    return logging.getLogger(name)


def setup_logging_from_config(name: str, cfg: dict[str, Any], log_key: str) -> logging.Logger:
    """Convenience wrapper: build the logger from the ``logging`` section of the config.

    ``log_key`` is ``"service_log"`` or ``"training_log"``.
    """
    lc = cfg["logging"]
    return setup_logging(
        name,
        log_file=Path(lc["dir"]) / lc[log_key],
        level=lc["level"],
        max_bytes=lc["max_bytes"],
        backup_count=lc["backup_count"],
    )
