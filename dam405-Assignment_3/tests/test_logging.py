"""Tests for src/utils/logging_config.py."""

from __future__ import annotations

import json
from pathlib import Path

from src.utils.logging_config import setup_logging


def _read_lines(path: Path) -> list[dict]:
    # StreamHandler (and RotatingFileHandler) flush after every record.
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def test_writes_json_lines(tmp_path: Path) -> None:
    log_file = tmp_path / "nested" / "test.log"
    logger = setup_logging("test.json", log_file=log_file, level="DEBUG")

    logger.debug("debug detail")
    logger.info("request served", extra={"request_id": "abc-123", "latency_ms": 4.2})

    records = _read_lines(log_file)
    assert len(records) == 2
    first, second = records
    for key in ("timestamp", "level", "logger", "module", "function", "line", "message"):
        assert key in second
    assert first["level"] == "DEBUG"  # file handler accepts DEBUG
    assert second["message"] == "request served"
    assert second["request_id"] == "abc-123"
    assert second["latency_ms"] == 4.2
    assert second["logger"] == "test.json"
    assert second["timestamp"].endswith("+00:00")


def test_exception_is_logged(tmp_path: Path) -> None:
    log_file = tmp_path / "err.log"
    logger = setup_logging("test.exc", log_file=log_file)
    try:
        1 / 0
    except ZeroDivisionError:
        logger.exception("boom")
    (record,) = _read_lines(log_file)
    assert record["level"] == "ERROR"
    assert "ZeroDivisionError" in record["exc_info"]


def test_no_duplicate_handlers(tmp_path: Path) -> None:
    log_file = tmp_path / "dup.log"
    setup_logging("test.dup", log_file=log_file)
    logger = setup_logging("test.dup", log_file=log_file)
    assert len(logger.handlers) == 2  # console + file
    logger.info("once")
    assert len(_read_lines(log_file)) == 1
