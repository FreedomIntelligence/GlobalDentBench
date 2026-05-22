# Created: 2026-02-01
# Modified: 2026-03-20
# Purpose: Provide lightweight configuration helpers for shared utility modules.

import json
from pathlib import Path
from typing import Any, Dict, Optional


PROJECT_ROOT = Path(__file__).parent.parent
CONFIG_FILE = PROJECT_ROOT / "config" / "config.json"


def load_config(config_path: Optional[Path] = None) -> Dict[str, Any]:
    config_file = config_path or CONFIG_FILE
    if not config_file.exists():
        raise FileNotFoundError(f"Config file not found: {config_file}")

    with open(config_file, "r", encoding="utf-8") as file:
        return json.load(file)


def get_model_config(config_type: str, config_path: Optional[Path] = None) -> Dict[str, Any]:
    config = load_config(config_path)
    if config_type not in config:
        raise KeyError(f"Config section `{config_type}` not found in config.json.")
    return config[config_type]


def get_llm_config() -> Dict[str, Any]:
    return get_model_config("llm")


def get_vlm_config() -> Dict[str, Any]:
    return get_model_config("vlm")


from . import image_process, llm_api


__all__ = [
    "load_config",
    "get_model_config",
    "get_llm_config",
    "get_vlm_config",
    "PROJECT_ROOT",
    "llm_api",
    "image_process",
]
