# src/modeling.py
from __future__ import annotations

"""
This is the modeling stage for the Telco churn pipeline

Responsibilities
----------------
- Building sklearn Pipelines that include both:
    * the preprocessing ColumnTransformer (unfitted when passed in)
    * the models for estimating (LogReg + Random Forest)

- Fitting the pipelines only on the training data:
    model_pipeline.fit(X_train, y_train)
    
Key Principles
--------------
- No leakage:
    The preprocessor is not fit in preprocessing.py.
    It is fit inside model_pipeline.fit(X_train, y_train) only.
- Deployable artifacts:
    The fitted sklearn Pipeline can be saved via joblib and used for inference
    as a single object (preprocessing plus the model)
"""

from dataclasses import dataclass
from typing import Dict

from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from . import config


@dataclass(frozen = True)
class TrainedModel:
    """
    This is a small wrapper that keeps the trained pipelines and its name together,
    which makes saving the evaluation/artifacts easier and more explicit.
    """
    name: str
    pipeline: Pipeline


def train_models(preprocessor, X_train, y_train) -> Dict[str, TrainedModel]:
    """
    This function trains selected models based on their configuration flags.
    
    Parameters
    ----------
    preprocessor: 
        This is an unfitted ColumnTransformer (from preprocessing.build_processor()).
    X_train, y_train:
        This is the training split. Only this data will be used for the fitting.
        
    Returns
    -------
    Dict[str, TrainedModel]
        This is a dictionary where the model name is the key (i.e. "logreg", "random_forest"),
        where each value contains the fitted sklearn Pipeline.
    """
    trained: Dict[str, TrainedModel] = {}
    
    if config.USE_LOGREG:
        # This is the baseline linear model: Logistic Regression
        # class_weight = "balanced", and this helps with class imbalance 
        estimator = LogisticRegression(
            max_iter = 2000,
            class_weight = "balanced",
            random_state = config.RANDOM_SEED,
        )
    
        pipeline = Pipeline(
            steps = [
                ("preprocessor", preprocessor),
                ("model", estimator),
            ]
        )
        
        pipeline.fit(X_train, y_train)
        trained["logreg"] = TrainedModel(name = "logreg", pipeline = pipeline)
    
    if config.USE_RANDOM_FOREST:
        # This is the non-linear comparison model: Random Forest
        estimator = RandomForestClassifier(
            n_estimators = config.RF_N_ESTIMATORS,
            max_depth = getattr(config, "RF_MAX_DEPTH", None),
            random_state = config.RANDOM_SEED,
            class_weight = "balanced_subsample",
            n_jobs = 1,
        )
    
        pipeline = Pipeline(
            steps = [
                ("preprocessor", preprocessor),
                ("model", estimator),
            ]
        )
        
        pipeline.fit(X_train, y_train)
        trained["random_forest"] = TrainedModel(name = "random_forest", pipeline = pipeline)
        
    if not trained:
        raise ValueError(
            "No models were trained. Check config.USE_LOGREG and config.USE_RANDOM_FOREST."
        )
    
    return trained
    