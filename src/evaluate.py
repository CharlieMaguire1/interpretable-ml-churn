# src/evaluate.py
from __future__ import annotations

"""
This file is for evaluation and artefact stage for the Telco churn pipeline

Responsibilities
----------------
- Select a decision threshold using the validation set:
    * For each threshold in config.thresholds:
        - compute F1 on the validation predictions
    * choose the threshold that maximises validation F1
    
- Evaluate on the test set using the current threshold:
    - ROC-AUC
    - PR-AUC
    - Accuracy
    - Precision, Recall, F1
    - Confusion matrix

- Save the artefacts:
    - metrics JSON (outputs/metrics/)
    - confusion matrix PNG (outputs/figures/)
    - model pipeline joblib (outputs/models/)
    
Key Principles
--------------
- No leakage
    the threshold selections is done on the validation set and not the test
- Deterministic:
    the thresholds are fixed from config.thresholds
- Traceability:
    the artefacts are named with the model name, and the metrics record the 
    chosen threshold
"""

import json
from dataclasses import asdict, dataclass
from typing import Dict, Tuple

import numpy as np
import matplotlib.pyplot as plt
from joblib import dump
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    average_precision_score
)

from .paths import ensure_directories, models_dir, metrics_dir, figures_dir
from . import config
from .modeling import TrainedModel


@dataclass (frozen = True)
class EvaluationResult:
    """
    This class captures the final evaluation outputs for a trained model.
    Then this is saved to JSON for reproducibility and reporting.
    """
    model_name: str
    chosen_threshold: float
    
    roc_auc: float
    pr_auc: float
    
    accuracy: float
    precision: float
    recall: float
    f1: float
    
    confusion_matrix: Tuple[Tuple[int, int], Tuple[int, int]]
    

def _predict_proba_positive(model_pipeline, X) -> np.ndarray:
    """
    This function obtains P(y = 1) from a fitted sklearn Pipeline

    Notes
    -----
    - For classifiers with predict_proba, use [:, 1]
    - If predict_proba is not available then this will cause an error
    """
    if not hasattr(model_pipeline, "predict_proba"):
        raise AttributeError(
            "Model pipeline does not support predict_proba. "
            "The expectation for this pipeline are classifier with probability outputs"
        )
    proba = model_pipeline.predict_proba(X)[:, 1]
    return proba


def _select_threshold_by_f1(
    y_true: np.ndarray,
    y_proba: np.ndarray
) -> Tuple[float, Dict[float, float]]:
    """
    This function sweeps through the thresholds and chooses the one that maximises F1
    
    Returns
    -------
    best_threshold: float
    f1_by_threshold: Dict[float, float]
        This is a mapping for traceability/debugging
    """
    f1_by_threshold: Dict[float, float] = {}
    
    best_threshold = float(config.DEFAULT_THRESHOLD)
    best_f1 = -1.0
    
    for t in config.THRESHOLDS:
        y_pred = (y_proba >= t).astype(int)
        score = f1_score(y_true, y_pred)
        f1_by_threshold[float(t)] = float(score)
        
        if score > best_f1:
            best_f1 = score
            best_threshold = float(t)
            
    return best_threshold, f1_by_threshold


def _plot_confusion_matrix(cm: np.ndarray, title: str, path) -> None:
    """
    This function saves a simple confusion matrix to the disk
    """
    fig, ax = plt.subplots()
    ax.imshow(cm)
    
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["No churn", "Churn"])
    ax.set_yticklabels(["No churn", "Churn"])
    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    
    # Adding the numbers
    for (i, j), value in np.ndenumerate(cm):
        ax.text(j, i, str(value), ha = "center", va = "center")
    
    fig.tight_layout()
    fig.savefig(path, dpi = 100)
    plt.close(fig)
    

def evaluate_and_save(
    trained_models: Dict[str, TrainedModel],
    *,
    X_val,
    y_val,
    X_test,
    y_test,
) -> Dict[str, EvaluationResult]:
    """
    This functions evaluates each trained model and saves the artefacts.
    
    Parameters
    ---------
    trained_models:
        This is a Dict of the model name -> A TrainedModel wrapper
    X_val, y_val: 
        This is the validation split for the threshold selection
    X_test, y_test:
        This is the test split for the final metrics, which is evaluated once with the
        chosen threshold
        
    Returns
    -------
    Dict[str, EvaluationResult]
        The evaluation results for each model
    """
    ensure_directories()
    
    results: Dict[str, EvaluationResult] = {}
    
    for name, tm in trained_models.items():
        pipeline = tm.pipeline
        
        # --- Threshold selection for Validation ---
        val_proba = _predict_proba_positive(pipeline, X_val)
        best_t, f1_sweep = _select_threshold_by_f1(y_val.to_numpy(), val_proba)
        
        # --- Final Evaluation for Test ---
        test_proba = _predict_proba_positive(pipeline, X_test)
        test_pred = (test_proba >= best_t).astype(int)
        
        y_test_arr = y_test.to_numpy()
        roc_auc = roc_auc_score(y_test_arr, test_proba)
        pr_auc = average_precision_score(y_test_arr, test_proba)

        acc = accuracy_score(y_test_arr, test_pred)
        prec = precision_score(y_test_arr, test_pred, zero_division = 0)
        rec = recall_score(y_test_arr, test_pred, zero_division = 0)
        f1 = f1_score(y_test_arr, test_pred, zero_division = 0)
        
        cm = confusion_matrix(y_test_arr, test_pred)
        cm_tuple = ((int(cm[0, 0]), int(cm[0, 1])), (int(cm[1, 0]), int(cm[1, 1])))
        
        result = EvaluationResult(
            model_name = name,
            chosen_threshold = float(best_t),
            roc_auc = float(roc_auc),
            pr_auc = float(pr_auc),
            accuracy = float(acc),
            precision = float(prec),
            recall = float(rec),
            f1 = float(f1),
            confusion_matrix = cm_tuple,
            )
        results[name] = result 
        
        # --- Save the model artifact ---
        model_path = models_dir / f"{name}.joblib"
        dump(pipeline, model_path)
        
        # --- Save the metrics in JSON including the sweep for traceability ---
        metrics_payload = {
            "result": asdict(result),
            "threshold_sweep_f1": f1_sweep,
            "config": {
                "random_seed": config.RANDOM_SEED,
                "val_test_size": config.VAL_TEST_SIZE,
                "test_size": config.TEST_SIZE,
                "primary_metric": config.PRIMARY_METRIC,
            },
        }
        
        metrics_path = metrics_dir / f"{name}_metrics.json"
        with open(metrics_path, "w", encoding = "utf-8") as f:
            json.dump(metrics_payload, f, indent = 2)
            
        # --- Save confusion matrix figure ---
        fig_path = figures_dir / f"{name}_confusion_matrix.png"
        _plot_confusion_matrix(cm, title = f"Confusion Matrix - {name}", path = fig_path)
     
    return results