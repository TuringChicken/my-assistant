"""Load configuration from config/settings.yaml, with .env overrides."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

# Project root = parent of the `assistant` package directory.
ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SETTINGS = ROOT / "config" / "settings.yaml"


def load_settings(path: str | Path = DEFAULT_SETTINGS) -> dict[str, Any]:
    """Read YAML settings and apply environment overrides."""
    load_dotenv(ROOT / ".env")

    with open(path, encoding="utf-8") as f:
        settings: dict[str, Any] = yaml.safe_load(f)

    # Let the environment override the Ollama host (handy for Docker, etc.).
    host = os.getenv("OLLAMA_HOST")
    if host:
        settings["ollama"]["host"] = host

    return settings
