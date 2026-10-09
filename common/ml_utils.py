"""Shared utilities for the Epic 5 machine learning pipeline."""

import pandas as pd

from epic4_ml.day17_feature_engineering import (
    add_engineered_features,
    build_preprocessor,
)


CLASSIFICATION_DATA = "epic4_ml/data/breast_cancer.csv"
REGRESSION_DATA = "epic4_ml/data/california_housing.csv"


def load_dataset(task: str):
    """Load the dataset and separate features and target."""
    if task == "classification":
        df = pd.read_csv(CLASSIFICATION_DATA)
        X = df.drop(columns=["target"])
        y = df["target"]

    elif task == "regression":
        df = pd.read_csv(REGRESSION_DATA)
        X = df.drop(columns=["MedHouseVal"])
        y = df["MedHouseVal"]
        X = add_engineered_features(X)

    else:
        raise ValueError(
            "task must be 'classification' or 'regression'"
        )

    return X, y


def build_day17_preprocessor(X: pd.DataFrame):
    """Build the existing Day 17 preprocessing pipeline."""
    numeric_cols = X.select_dtypes(
        include=["number"]
    ).columns.tolist()

    categorical_cols = X.select_dtypes(
        include=["object", "category"]
    ).columns.tolist()

    return build_preprocessor(
        numeric_cols=numeric_cols,
        categorical_cols=categorical_cols,
    )