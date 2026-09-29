from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    MinMaxScaler,
    OneHotEncoder,
    OrdinalEncoder,
    StandardScaler,
)

from epic4_ml.day16_ml_basics import (
    california_X_train,
    california_X_test,
    california_y_train,
    california_y_test,
)


BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "california_housing.csv"
MODEL_PATH = BASE_DIR / "models" / "preprocessor.joblib"


def build_preprocessor(
    numeric_cols: list, categorical_cols: list
) -> ColumnTransformer:
    """Build reusable preprocessing pipeline for numeric and categorical columns."""

    minmax_cols = [
        col
        for col in numeric_cols
        if col in {"Rooms_per_Occupant", "Bedroom_Room_Ratio"}
    ]

    standard_cols = [
        col for col in numeric_cols if col not in minmax_cols
    ]

    standard_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    minmax_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", MinMaxScaler()),
        ]
    )

    nominal_cols = [
        col for col in categorical_cols if col != "HouseAge_Bin"
    ]

    ordinal_cols = [
        col for col in categorical_cols if col == "HouseAge_Bin"
    ]

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
            ),
        ]
    )

    ordinal_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OrdinalEncoder(
                    categories=[["Young", "Adult", "Mature", "Old"]],
                    handle_unknown="use_encoded_value",
                    unknown_value=-1,
                ),
            ),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("numeric", standard_pipeline, standard_cols),
            ("minmax_numeric", minmax_pipeline, minmax_cols),
            ("categorical", categorical_pipeline, nominal_cols),
            ("ordinal", ordinal_pipeline, ordinal_cols),
        ],
        remainder="drop",
    )


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add new engineered features without modifying the input DataFrame."""

    result = df.copy()

    # Ratio feature: average rooms per occupant.
    if {"AveRooms", "AveOccup"}.issubset(result.columns):
        result["Rooms_per_Occupant"] = (
            result["AveRooms"] / result["AveOccup"].replace(0, np.nan)
        )

    # Ratio feature: bedrooms relative to rooms.
    if {"AveBedrms", "AveRooms"}.issubset(result.columns):
        result["Bedroom_Room_Ratio"] = (
            result["AveBedrms"] / result["AveRooms"].replace(0, np.nan)
        )

    # Interaction feature: income combined with house age.
    if {"MedInc", "HouseAge"}.issubset(result.columns):
        result["Income_Age_Interaction"] = (
            result["MedInc"] * result["HouseAge"]
        )

    # Ordered categorical feature: house age groups.
    if "HouseAge" in result.columns:
        result["HouseAge_Bin"] = pd.cut(
            result["HouseAge"],
            bins=[-np.inf, 10, 20, 40, np.inf],
            labels=["Young", "Adult", "Mature", "Old"],
        ).astype(object)

    return result


def select_top_features(X, y, k: int) -> list:
    """Return the names of the top k features using f_regression."""

    selector = SelectKBest(score_func=f_regression, k=k)
    selector.fit(X, y)

    if hasattr(X, "columns"):
        return X.columns[selector.get_support()].tolist()

    return np.flatnonzero(selector.get_support()).tolist()


def get_feature_names(preprocessor) -> list:
    """Return readable names for transformed features."""

    return preprocessor.get_feature_names_out().tolist()


def main():
    df = pd.read_csv(DATA_PATH)

    target = "MedHouseVal"
    X = df.drop(columns=[target])
    y = df[target]

    X = add_engineered_features(X)

    # Reuse the train/test split created by Day16.
    X_train = X.loc[california_X_train.index]
    X_test = X.loc[california_X_test.index]
    y_train = y.loc[california_y_train.index]
    y_test = y.loc[california_y_test.index]

    numeric_cols = X_train.select_dtypes(
        include=["number"]
    ).columns.tolist()

    categorical_cols = X_train.select_dtypes(
        include=["object", "category"]
    ).columns.tolist()

    preprocessor = build_preprocessor(
        numeric_cols=numeric_cols,
        categorical_cols=categorical_cols,
    )

    # Fit only on training data.
    transformed_train = preprocessor.fit_transform(X_train)

    # Transform test data using the fitted training preprocessor.
    transformed_test = preprocessor.transform(X_test)

    print("Before preprocessing:")
    print("Train shape:", X_train.shape)
    print("Test shape:", X_test.shape)

    print("\nAfter preprocessing:")
    print("Train shape:", transformed_train.shape)
    print("Test shape:", transformed_test.shape)

    feature_names = get_feature_names(preprocessor)

    print("\nColumn-name comparison:")
    print("Before:", X_train.columns.tolist())
    print("After:", feature_names)

    print("\nNaN check:")
    print("Train NaNs:", int(np.isnan(transformed_train).sum()))
    print("Test NaNs:", int(np.isnan(transformed_test).sum()))

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(preprocessor, MODEL_PATH)

    loaded_preprocessor = joblib.load(MODEL_PATH)
    reloaded_train = loaded_preprocessor.transform(X_train)

    print("\nReload verification:")
    print(
        "Identical output:",
        np.allclose(transformed_train, reloaded_train),
    )

    transformed_train_df = pd.DataFrame(
        transformed_train,
        columns=feature_names,
    )

    top_features = select_top_features(
        transformed_train_df,
        y_train,
        k=5,
    )

    print("\nTop 5 features:")
    print(top_features)


if __name__ == "__main__":
    main()