# src/preprocessing.py
from __future__ import annotations

"""
This is the preprocessing stage for the Telco churn pipeline

Where this file sits in the pipeline
-----------------------------------
1. Ingestion (src/ingestion.py)
    - load_raw_data() reads the raw Telco CSV from src/paths.raw_data_path
    - and then attaches the provenance columns:
        __source_file, __ingested_at_utc
    
2. Preprocessing (this module)
    This module provides the deterministic, reusable steps used by run_pipeline.py at the root.
    
    - clean_total_charges(data): 
        This fixes a known dataset quirk where TotalCharges could be stored as strings or blanks.
        It also converts to numeric and drops the invalid rows (part of the deterministic strategy)
        
    - make_xy(data): 
        This enforces the feature/target contract, whereby:
            * target = Churn mapped {"No":0, "Yes":1} 
            * drop identifier (customerID)
            * drop provenance metadata (__source_file, __ingested_at_utc), if they are present
    
    - infer_feature_types(X): 
        This infers the numeric vs categorical columns for the ColumnTransformer.
        And includes a domain rule: SeniorCitizen is treated as categorical.
    
    - split_data(X, y): 
        This creates the stratified train/val/test splits using config.* ratios and 
        config.RANDOM_SEED, making it reproducible.
    
    - build_preprocessor(numeric_cols, categorical_cols): 
        This returns an unfitted sklearn ColumnTransformer:
            * numeric: median impute + standard scale
            * categorical: most_frequent impute + one-hot (handle_unknown = "ignore")
    
    - prepare_training_data(data): 
        This is the pipeline wrapper that ties everything together
        
3. Modeling (src/modeling.py)
    - train_models(preprocessor, X_train, y_train) builds an sklearn Pipeline:
        Pipeline([("preprocessor", preprocessor), ("model", ...)])
    - IMPORTANT: the preprocessor is fit during model.fit(X_train, y_train),
      so the preprocessing statistics are learned from the training data only (no leakage).
      
4. Evaluation and Artifacts (src/evaluate.py, src/paths.py)
    - evaluate_models(...) select a decision threshold on the validation data (max F1),
      evaluates on the test data with that threshold, and writes to:
        output/metrics/*, outputs/figures/*
    - trained pipelines are saved with joblib to:
        output/models/*
        
Key Principles
-------------
- Deterministic & reproducible:
    the behaviour is controlled by src/config.py (RANDOM_SEED, split ratios, thresholds, 
    model flags).
- No leakage:
    the split happens before the fitting: preprocessor is only fit within training .fit().
- Traceability:
    the ingestion adds provenance, and preprocessing drops provenance from the features.
- Deployable artifacts:
    they are saved joblib pipelines that include both the fitted preprocessor and the fitted
    model.    
"""

import pandas as pd

from dataclasses import dataclass
from typing import List, Tuple

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

from . import config


# ---------------------------
# Data contracts and constraints
# ---------------------------

TARGET_COL = "Churn"
ID_COL = "customerID"

# These are the provenance fields added in ingestion.py
PROVENANCE_COLS = ["__source_file", "__ingested_at_utc"]


@dataclass(frozen = True)
class TrainingData:
    """
    This is a structured return object for the preprocessing stage.
    
    This exists because:
    - It avoids passing around too many tuples, which improves readability.
    - It makes the pipeline stages explicit, meaning that it is traceable.
    - It enforces a contract that is stable between preprocessing and modeling. 
    """
    
    X_train: pd.DataFrame
    X_val: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_val: pd.Series
    y_test: pd.Series
    
    preprocessor: ColumnTransformer
    numeric_cols: List[str]
    categorical_cols: List[str]
    
    
# ----------------------------
# Step 1: Dataset Fixes
# ----------------------------

def clean_total_charges(data: pd.DataFrame) -> pd.DataFrame:
    """
    This function fixes the known quirk found in Telco Churn dataset, which is:
    - The TotalCharges column can contain blanks and it is stored as a string sometimes
    - So it has to be coerced to numeric data type, and invalid rows will become NaN
    - And then those rows will be dropped

    Deterministic decision:
    - We drop the invalid rows rather than imputing them because that would cause a data issue
    - And dropping is a deterministic and an explainable action.
    """
    df = data.copy()
    
    if "TotalCharges" not in df.columns:
        # So if the dataset schema changes, it will cause an error
        raise KeyError("The expected column 'TotalCharges' is not found in the input data")
    
    # Conversion to numeric, which is blanks/strings to NaN
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors = "coerce")
    
    # Dropping the rows where TotalCharges could not be parsed
    before_parse = len(df)
    df = df.dropna(subset = ["TotalCharges"]).reset_index(drop = True)
    after_parse = len(df)
    
    # Uncomment below if you need to do add logging later 
    # dropping = before_parse - after_parse
    
    return df


# --------------------------------
# Step 2: Feature/target contract
# --------------------------------

def make_xy(data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """
    This functions splits the dataset into features X and target y, whilst maintaining a 
    stable contract
    
    - Target:
        Churn mapped {"No": 0, "Yes": 1}
    - Drop:
        * customerID (identifier)
        * provenance metadata (__source_file, __ingested_at_utc) if present
        
    Returns
    -------
    X: pd.DataFrame
        A feature matrix with no identifier, no provenance, and no target
    y: pd.Series
        This is a binary target series that is aligned with X
    """
    df = data.copy()
    
    if TARGET_COL not in df.columns:
        raise KeyError(f"The expected target column '{TARGET_COL}' not found in the input data.")
    
    # Mapping the churn label deterministically
    y = df[TARGET_COL].map({"No": 0, "Yes": 1})
    
    if y.isna().any():
        # If unexpected labels appear then it will cause an error
        # bad_values stores all the unique unexpected values extracted from the Churn column that
        # failed mapping
        bad_values = sorted(df.loc[y.isna(), TARGET_COL].unique().tolist())
        raise ValueError(
            f"There are unexpected values in target column '{TARGET_COL}': {bad_values}."
            "Expected only 'No'/'Yes'."
        )
        
    # Constructing X by dropping the target, ID, and the provenance
    drop_cols = [TARGET_COL]
    
    if ID_COL in df.columns:
        drop_cols.append(ID_COL)
        
    for c in PROVENANCE_COLS:
        if c in df.columns:
            drop_cols.append(c)
            
    X = df.drop(columns = drop_cols) 
    
    return X, y


# ------------------------------------
# Step 3: Infer Feature Types
# ------------------------------------

def infer_feature_types(X: pd.DataFrame) -> Tuple[List[str], List[str]]:
    """
    Infer the numeric vs categorical feature columns for sklearn ColumnTransformer
    
    Domain rule:
    - The SeniorCitizen column is treated as categorical even though it may be a 0/1 integer
    
    Returns
    --------
    numeric_cols: List[str]
    categorical_cols: List[str]
    """
    if not isinstance(X, pd.DataFrame):
        raise TypeError("X must be a pandas DataFrame.")
    
    numeric_cols = X.select_dtypes(include = "number").columns.tolist()
    categorical_cols = X.select_dtypes(exclude = "number").columns.tolist()
    
    # Domain rule: treat SeniorCitizen as categorical
    if "SeniorCitizen" in numeric_cols:
        numeric_cols.remove("SeniorCitizen")
        categorical_cols.append("SeniorCitizen")
        
    # Deterministic ordering, which is useful for reproducible pipelines and debugging.
    numeric_cols = sorted(numeric_cols)
    categorical_cols = sorted(categorical_cols)
    
    return numeric_cols, categorical_cols


# ----------------------------------------
# Step 4: Deterministic Splits
# ----------------------------------------

def split_data(
    X: pd.DataFrame,
    y: pd.Series,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """
    This function creates a deterministic stratified train/val/test splits.7
    
    Expected outcome on Telco dataset after cleaning:
    - ~70/15/15 split with churn rate ~0.266 preserved across splits.

    Configuration behaviour (from config.py):
    - VAL_TEST_SIZE: A fraction of the dataset to reserve for val and test
    - TEST_SIZE: A fraction of that reserved part that goes to the test
    (for example: VAL_TEST_SIZE = 0.3 and TEST_SIZE = 0.5 -> 15% val, 15% test)
    
    Returns
    -------
    X_train, X_val, X_test, y_train, y_val, y_test
    """
    # The first split between train and test (val + test)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size = config.VAL_TEST_SIZE, 
        random_state = config.RANDOM_SEED, stratify = y
    )
    
    # The second split between val and test
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size = config.TEST_SIZE,
        random_state = config.RANDOM_SEED, stratify = y_temp
    )
    
    return X_train, X_val, X_test, y_train, y_val, y_test

# --------------------------------------
# Step 5: Construction of unfitted preprocessor
# --------------------------------------

def build_preprocessor(
    numeric_cols: List[str],
    categorical_cols: List[str],
) -> ColumnTransformer:
    """
    This function constructs an unfitted ColumnTransformer for numeric and 
    categorical processing
    
    Numeric:
    - median imputation
    - standard scaling
    
    Categorical: 
    - most_frequent imputation
    - one-hot encoding (handle_unknown = 'ignore')
    
    IMPORTANT:
    - This preprocessor is not fitted here
    - It should only be fitted only within the model.fit(X_train, y_train)
      to avoid leakage
    """
    numeric_transformer = Pipeline(
        steps = [
            ("imputer", SimpleImputer(strategy = "median")),
            ("scaler", StandardScaler()),
        ]
    )
    
    categorical_transformer = Pipeline(
        steps = [
            ("imputer", SimpleImputer(strategy = "most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown = "ignore")),
        ]
    )
    
    preprocessor = ColumnTransformer(
        transformers = [
            ("num", numeric_transformer, numeric_cols),
            ("cat", categorical_transformer, categorical_cols),
        ],
        remainder = "drop",
        verbose_feature_names_out = False
    )
    
    return preprocessor

# -------------------------------------
# This part wraps everything together
# ------------------------------------

def prepare_training_data(data: pd.DataFrame) -> TrainingData:
    """
    This function is the Pipeline wrapper for the preprocessing
    
    Responsibilities:
    - Applying the dataset fixes (clean_total_charges)
    - Enforcing the feature/target contract (make_xy)
    - Inferring the feature types (infer_feature_types)
    - Deterministically split into train/val/test (split_data)
    - Building an unfitted preprocessor (build_preprocessor)
    
    Returns
    -------
    TrainingData
        A structured object that contains the following:
        - train/val/test splits
        - an unfitted preprocessor
        - inferred feature type lists
    """
    # 1. Cleaning according to the dataset (deterministic)
    cleaned = clean_total_charges(data)
    
    # 2. Enforcing the contracts
    X, y = make_xy(cleaned)
    
    # 3. Inferring the feature types
    numeric_cols, categorical_cols = infer_feature_types(X)
    
    # 4. Deterministic stratified splits
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)
    
    # 5. Unfitted preprocessor (the fit happens during model training)
    preprocessor = build_preprocessor(numeric_cols, categorical_cols)
    
    return TrainingData(
        X_train = X_train,
        X_val = X_val,
        X_test = X_test,
        y_train = y_train,
        y_val = y_val,
        y_test = y_test,
        preprocessor = preprocessor,
        numeric_cols = numeric_cols,
        categorical_cols = categorical_cols,
    )
    
