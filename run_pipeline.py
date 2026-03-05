# run_pipeline.py
from __future__ import annotations

"""
This script is the entry point for the Telco churn pipeline.

This file stays small intentionally and is for orchestration only.
No data transformations or model logic should be in this script.

Pipeline flow
-------------
1. Ensure the directories exist (src/paths.py)
2. Ingestion: load_raw_data() adds the provenance metadata
3. Preprocessing: prepare_training_data() performs deterministic cleaning, contracts,
   splits, and builds an unfitted preprocessor
4. Modeling: train_models() fits the sklearn pipeline on the train split only (no leakage)
5. Evaluation: evaluate_and_save() selects the threshold from the validation set,
   evaluates on the test set, and then saves the artifacts (models, metrics, and figures)
"""

from src.paths import ensure_directories
from src.ingestion import load_raw_data
from src.preprocessing import prepare_training_data
from src.modeling import train_models
from src.evaluate import evaluate_and_save


def main() -> None:
    # 0. Creating the output directories, which is safe to call and idempotent
    ensure_directories()
    
    # 1. Ingestion
    # Reading the raw dataset and attaching the provenance columns for traceability
    # - __source_file, and __ingested_at_utc
    data = load_raw_data()
    
    # 2. Preprocessing 
    # Applying the deterministic cleaning,
    # Enforcing the feature/target contract, and infers feature types,
    # And produces reproducible train/val/test splits.
    # Also constructs an unfitted ColumnTransformer (fit is done with train split).
    td = prepare_training_data(data)
    
    # 3. Modeling (only fit on the train split)
    # Building the sklearn Pipelines of the following form:
    #   Pipeline([("preprocessor", preprocessor), ("model", estimator)])
    # The preprocessor is fit within pipeline.fit(X_train, y_train) to avoid leakage.
    trained = train_models(td.preprocessor, td.X_train, td.y_train)
    
    # 4. Evaluation and saving the artifacts
    # - Selecting a decision threshold using the validation set (e.g., max F1)
    # - Evaluating the final metrics on the test set only once with the best threshold
    # - Writing the artifacts to outputs/:
    #       outputs/models/*.joblib
    #       outputs/metrics/*_metrics.json
    #       outputs/figures/*_confusion_matrix.png
    results = evaluate_and_save(
        trained,
        X_val = td.X_val,
        y_val = td.y_val,
        X_test = td.X_test,
        y_test = td.y_test,
    )
    
    # 5. Minimal summary of the console 
    # This is intentionally lighweight as it is a quick sanity check after a run.
    print("\n=== Pipeline complete ===")
    for name, res in results.items():
        print(
            f"- {name}: ROC-AUC = {res.roc_auc:.4f}, PR-AUC = {res.pr_auc:.4f}, "
            f"F1 = {res.f1:.4f} (threshold = {res.chosen_threshold:.2f})"
        )

if __name__ == "__main__":
    # This is the entry point so that the imports do not accidentally execute the pipeline.
    main()  