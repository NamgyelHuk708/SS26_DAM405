"""Configuration loading for the project.

Every module reads its settings (paths, seeds, thresholds) from
``config/config.yaml`` through :func:`load_config`. Relative paths in the
config are interpreted relative to :data:`PROJECT_ROOT`; use
:func:`resolve_path` to turn them into absolute ``Path`` objects.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH: Path = PROJECT_ROOT / "config" / "config.yaml"


@lru_cache(maxsize=None)
def _load_cached(config_path: str) -> dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh) or {}
    if not isinstance(cfg, dict):
        raise ValueError(f"Config file {config_path} must contain a mapping at top level.")
    return cfg


def load_config(config_path: str | Path | None = None) -> dict[str, Any]:
    """Load the YAML config and return it as a dict.

    The path may be overridden with the ``DAM405_CONFIG`` environment variable
    (useful for tests). The parsed file is cached; a shallow copy is returned so
    callers can't mutate the cache's top-level keys by accident.
    """
    path = Path(config_path or os.environ.get("DAM405_CONFIG", DEFAULT_CONFIG_PATH))
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    return dict(_load_cached(str(path)))


def resolve_path(relative: str | Path) -> Path:
    """Return ``relative`` as an absolute path anchored at :data:`PROJECT_ROOT`."""
    p = Path(relative)
    return p if p.is_absolute() else PROJECT_ROOT / p


def mlflow_tracking_uri(cfg: dict[str, Any]) -> str:
    """Return the MLflow tracking URI, anchoring relative SQLite paths at PROJECT_ROOT.

    ``sqlite:///mlflow.db`` is relative to the current working directory, which
    would create a different database depending on where a script is launched.
    """
    uri: str = cfg["mlflow"]["tracking_uri"]
    prefix = "sqlite:///"
    if uri.startswith(prefix) and not uri[len(prefix):].startswith("/"):
        return prefix + str(resolve_path(uri[len(prefix):]))
    return uri


def mlflow_artifact_location(cfg: dict[str, Any]) -> str:
    """Return the experiment artifact location as an absolute ``file://`` URI."""
    return resolve_path(cfg["mlflow"]["artifact_location"]).as_uri()
