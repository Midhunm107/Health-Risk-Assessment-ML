"""Shared, leakage-free preprocessing used by every model in the project.

All three classifiers (Decision Tree, Random Forest, XGBoost) must be trained and
tested on exactly the same data, otherwise their results cannot be compared. This
module is the single place that defines that pipeline:

1. target  = equal-frequency (quantile) binning of ``heart_disease_risk_score``
             into three classes: Low / Medium / High
2. drop    = the five pre-computed risk-score columns (data leakage)
3. encode  = one-hot encode the categorical columns (28 features -> 40)
4. split   = stratified 80:20 train/test split, ``random_state = 42``
"""
from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

TARGET_SOURCE = "heart_disease_risk_score"
RISK = ["Low", "Medium", "High"]
RISK_TO_NUM = {"Low": 0, "Medium": 1, "High": 2}  # XGBoost needs integer labels
NUM_TO_RISK = {v: k for k, v in RISK_TO_NUM.items()}

# Pre-computed scores that would let a model "read the answer" (hypertension score
# correlates r = 0.80 with the target). They are removed before training.
LEAKAGE_COLS = [
    "diabetes_risk_score",
    "hypertension_risk_score",
    "obesity_risk_score",
    "cholesterol_risk_score",
    TARGET_SOURCE,
]
BINARY_COLS = ["family_history_diabetes", "hypertension_history", "cardiovascular_history"]

TEST_SIZE = 0.2
RANDOM_STATE = 42


def add_risk_target(df: pd.DataFrame) -> pd.Series:
    """Return the 3-class target (Low / Medium / High) as plain strings."""
    groups = pd.qcut(df[TARGET_SOURCE], q=3, labels=RISK)
    return groups.astype(str)


def make_preprocessor(cat_cols: list[str]) -> ColumnTransformer:
    """One-hot encode the categorical columns and pass the numeric ones through."""
    return ColumnTransformer(
        [("categorical", OneHotEncoder(handle_unknown="ignore"), cat_cols)],
        remainder="passthrough",
    )


def load_and_split(csv_path: str):
    """Read the dataset and return the shared split.

    Returns
    -------
    x : DataFrame        all 28 raw input features (no leakage columns)
    x_train, x_test      stratified 80:20 split of ``x``
    y_train, y_test      Series of "Low" / "Medium" / "High"
    cat_cols, num_cols   names of the categorical and numeric feature columns
    """
    df = pd.read_csv(csv_path)
    y = add_risk_target(df)
    x = df.drop(columns=LEAKAGE_COLS)

    num_cols = x.select_dtypes(include="number").columns.tolist()
    cat_cols = [c for c in x.columns if c not in num_cols]

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    return x, x_train, x_test, y_train, y_test, cat_cols, num_cols
