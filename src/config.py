"""
config.py - Global configuration loader
Every module in this project imports from here.
Never hardcode paths or settings anywhere else.
"""

import yaml
import os
from pathlib import Path

# Find the project root (one level up from src/)
PROJECT_ROOT = Path(__file__).parent.parent

# Path to config.yaml
CONFIG_PATH = PROJECT_ROOT / "config.yaml"


def load_config() -> dict:
    """Load and return the full config.yaml as a dictionary."""
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(
            f"config.yaml not found at {CONFIG_PATH}\n"
            f"Please copy config.example.yaml to config.yaml and edit your paths."
        )
    with open(CONFIG_PATH, "r") as f:
        return yaml.safe_load(f)


# Load config once at import time
config = load_config()

# Convenience accessors - use these in other modules
PATHS = config["paths"]
OLLAMA = config["ollama"]
MODELS = config["models"]
HARDWARE = config["hardware"]
RETRIEVAL = config["retrieval"]
INTERFACE = config["interface"]
OCR = config["ocr"]

# Ensure critical directories exist
Path(PATHS["temp_dir"]).mkdir(parents=True, exist_ok=True)
Path(PATHS["workspaces_root"]).mkdir(parents=True, exist_ok=True)
