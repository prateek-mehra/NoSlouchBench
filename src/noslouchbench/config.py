from __future__ import annotations

from pathlib import Path

import yaml


def load_model_config(config_path: Path, model_name: str) -> dict:
    if not config_path.exists():
        return {}

    with config_path.open("r", encoding="utf-8") as f:
        doc = yaml.safe_load(f) or {}

    models = doc.get("models", {})
    return models.get(model_name, {})

