# src/paths.py

"""
This file is for centralised filesystem paths using pathlib
So that it can work from any working directory
"""

from __future__ import annotations
from pathlib import Path 

# Project root
project_root = Path(__file__).resolve().parents[1]

# Data
data_dir = project_root / "data"
raw_data_path = data_dir / "raw" / "telco_churn.csv"
interim_dir = data_dir / "interim"
processed_dir = data_dir / "processed"

# Outputs/ artefacts
output_dir = project_root / "outputs"
models_dir = output_dir / "models"
metrics_dir = output_dir / "metrics"
figures_dir = output_dir / "figures"

# A helper function to ensure the existence of directories
def ensure_directories() -> None:
    """Create the expected directories if they do not exist"""
    for dir in [data_dir, interim_dir, processed_dir, output_dir, models_dir,
                metrics_dir, figures_dir]:
        dir.mkdir(parents = True, exist_ok = True)
    


