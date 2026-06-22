"""Configuration loading utilities."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml


def get_project_root() -> Path:
    """Return the repository root directory."""
    return Path(__file__).resolve().parents[2]


@lru_cache(maxsize=1)
def load_settings() -> dict[str, Any]:
    """Load global settings from config/settings.yaml."""
    path = get_project_root() / "config" / "settings.yaml"
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


@lru_cache(maxsize=1)
def load_exercises_config() -> dict[str, Any]:
    """Load exercise definitions from config/exercises.yaml."""
    path = get_project_root() / "config" / "exercises.yaml"
    if not path.exists():
        return {"exercises": {}, "folder_aliases": {}}
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {"exercises": {}, "folder_aliases": {}}


def resolve_data_dir(settings: dict[str, Any] | None = None) -> Path:
    """Resolve the dataset directory, honoring BIOVISION_DATA_DIR override."""
    settings = settings or load_settings()
    env_override = os.environ.get("BIOVISION_DATA_DIR")
    if env_override:
        return Path(env_override).expanduser().resolve()
    rel = settings.get("paths", {}).get("data_dir", "Data")
    return (get_project_root() / rel).resolve()


def get_path(key: str, default: str) -> Path:
    """Resolve a configured path relative to project root."""
    settings = load_settings()
    rel = settings.get("paths", {}).get(key, default)
    path = Path(rel)
    if path.is_absolute():
        return path
    return get_project_root() / path
