"""Project-root path helpers (cwd-independent, Streamlit Cloud safe)."""

from __future__ import annotations

from pathlib import Path

# src/ -> project root
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent

MODELS_DIR: Path = PROJECT_ROOT / "models"
DATA_DIR: Path = PROJECT_ROOT / "data"
REPORTS_DIR: Path = PROJECT_ROOT / "reports"

MODEL_PATH: Path = MODELS_DIR / "emi_model.pkl"
DATASET_PATH: Path = DATA_DIR / "emi_dataset.csv"


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path
