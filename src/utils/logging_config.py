"""Logging configuration for BioVision AI."""

from __future__ import annotations

import logging
from typing import Optional

from src.utils.config import load_settings


def setup_logging(name: Optional[str] = None, level: Optional[str] = None) -> logging.Logger:
    """Configure and return a module logger."""
    settings = load_settings()
    log_cfg = settings.get("logging", {})
    log_level = (level or log_cfg.get("level", "INFO")).upper()
    log_format = log_cfg.get(
        "format",
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    logging.basicConfig(level=getattr(logging, log_level, logging.INFO), format=log_format)
    return logging.getLogger(name or "biovision")
