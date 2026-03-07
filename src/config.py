# src/config.py

"""
This file contains the global configurations for behaviours of experiments and reproducibility
Including random seeds, data split ratios, evaluation behaviour and model selection flags. 
"""

from __future__ import annotations
import numpy as np

RANDOM_SEED: int = 42 # Reproducibility

# Data Split: train(70%), val(15%), test(15%)
VAL_TEST_SIZE: float = 0.30 
TEST_SIZE: float = 0.50

# Evaluation
PRIMARY_METRIC: str = "roc_auc"
DEFAULT_THRESHOLD: float = 0.50
THRESHOLDS = np.linspace(0.05, 0.95, 19) # Threshold sweep in evaluate.py

# Models
USE_LOGREG: bool = True
USE_RANDOM_FOREST: bool = True

# Random Forest defaults
RF_N_ESTIMATORS: int = 400
RF_MAX_DEPTH: int = None

