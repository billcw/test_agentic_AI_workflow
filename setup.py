"""
setup.py - First-run setup script for Local AI Document Assistant
Run this once after cloning the repo and creating your config.yaml

Usage: python setup.py
"""

import os
import sys
from pathlib import Path


def check_python_version():
    """Ensure Python 3.11+"""
    if sys.version_info < (3, 11):
        print(f"ERROR: Python 3.11+ required. You have {sys.version}")
        sys.exit(1)
    print(f"  Python {sys.version_info.major}.{sys.version_info.minor} OK")


def check_config():
    """Ensure config.yaml exists."""
    config_path = Path("config.yaml")
    example_path = Path("config.example.yaml")
    if not config_path.exists():
        if example_path.exists():
            print("  ERROR: config.yaml not found.")
            print("  Please copy config.example.yaml to config.yaml and edit your paths.")
        else:
            print("  ERROR: Neither config.yaml nor config.example.yaml found.")
            print("  Are you running this from the project root directory?")
        sys.exit(1)
    print("  config.yaml found OK")


def create_directories():
    """Create all required project directories."""
    from src.config import PATHS

    dirs = [
        PATHS["workspaces_root"],
        PATHS["temp_dir"],
        "workspaces",
        "src/ingestion",
        "src/storage",
        "src/retrieval",
        "src/agents",
        "src/memory",
        "src/projects",
        "src/api",
        "tests",
        "docs",
    ]

    for d in dirs:
        Path(d).mkdir(parents=True, exist_ok=True)
        print(f"  Created/verified: {d}")


def check_ollama():
    """Check if Ollama is reachable."""
    import urllib.request
    from src.config import OLLAMA

    url = OLLAMA["base_url"]
    try:
        urllib.request.urlopen(f"{url}/api/tags", timeout=5)
        print(f"  Ollama reachable at {url} OK")
    except Exception:
        print(f"  WARNING: Ollama not reachable at {url}")
        print("  Make sure Ollama is running: ollama serve")


def check_tesseract():
    """Check if Tesseract is installed."""
    from src.config import PATHS
    tesseract = PATHS["tesseract_path"]
    if Path(tesseract).exists():
        print(f"  Tesseract found at {tesseract} OK")
    else:
        print(f"  WARNING: Tesseract not found at {tesseract}")
        print("  Install with: sudo apt install tesseract-ocr")


def main():
    print("=" * 60)
    print("  Local AI Document Assistant - Setup")
    print("=" * 60)

    print("\n[1] Checking Python version...")
    check_python_version()

    print("\n[2] Checking config.yaml...")
    check_config()

    print("\n[3] Creating directories...")
    create_directories()

    print("\n[4] Checking Ollama...")
    check_ollama()

    print("\n[5] Checking Tesseract...")
    check_tesseract()

    print("\n" + "=" * 60)
    print("  Setup complete! You are ready to start building.")
    print("=" * 60)


if __name__ == "__main__":
    main()
