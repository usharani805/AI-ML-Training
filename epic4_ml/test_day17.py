import joblib
import numpy as np
import pandas as pd

from epic4_ml.day17_feature_engineering import (
    MODEL_PATH,
    add_engineered_features,
    build_preprocessor,
    get_feature_names,
    select_top_features,
)


DATA_PATH = "epic4_ml/data/california_housing.csv"


def get_test_data():
    df = pd.read_csv(DATA_PATH)
    X = df.drop(columns=["MedHouseVal"])
    y = df["MedHouseVal"]
    X = add_engineered_features(X)

    numeric_cols = X.select_dtypes(include=["number"]).columns.tolist()
    categorical_cols = X.select_dtypes(
        include=["object", "category"]
    ).columns.tolist()

    return X, y, numeric_cols, categorical_cols


def test_engineered_features_are_added():
    df = pd.read_csv(DATA_PATH).drop(columns=["MedHouseVal"])

    result = add_engineered_features(df)

    expected = {
        "Rooms_per_Occupant",
        "Bedroom_Room_Ratio",
        "Income_Age_Interaction",
        "HouseAge_Bin",
    }

    assert expected.issubset(result.columns)


def test_input_dataframe_is_unchanged():
    df = pd.read_csv(DATA_PATH).drop(columns=["MedHouseVal"])
    original = df.copy(deep=True)

    add_engineered_features(df)

    pd.testing.assert_frame_equal(df, original)


def test_preprocessor_has_numeric_and_categorical_transformers():
    _, _, numeric_cols, categorical_cols = get_test_data()

    preprocessor = build_preprocessor(numeric_cols, categorical_cols)

    transformer_names = [
        name for name, _, _ in preprocessor.transformers
    ]

    assert "numeric" in transformer_names
    assert "categorical" in transformer_names


def test_no_nans_after_transformation():
    X, y, numeric_cols, categorical_cols = get_test_data()

    preprocessor = build_preprocessor(numeric_cols, categorical_cols)

    X_train = X.iloc[:100]
    preprocessor.fit(X_train, y.iloc[:100])

    transformed = preprocessor.transform(X_train)

    assert not np.isnan(transformed).any()


def test_unseen_category_does_not_crash():
    X, y, numeric_cols, categorical_cols = get_test_data()

    preprocessor = build_preprocessor(numeric_cols, categorical_cols)

    X_train = X.iloc[:100].copy()
    X_test = X.iloc[100:110].copy()

    preprocessor.fit(X_train, y.iloc[:100])

    if categorical_cols:
        X_test[categorical_cols[0]] = "Unseen_Category"

    transformed = preprocessor.transform(X_test)

    assert transformed.shape[0] == len(X_test)


def test_missing_values_are_handled():
    X, y, numeric_cols, categorical_cols = get_test_data()

    X_train = X.iloc[:100].copy()

    if numeric_cols:
        X_train.loc[X_train.index[0], numeric_cols[0]] = np.nan

    if categorical_cols:
        X_train.loc[X_train.index[1], categorical_cols[0]] = np.nan

    preprocessor = build_preprocessor(numeric_cols, categorical_cols)

    transformed = preprocessor.fit_transform(X_train, y.iloc[:100])

    assert not np.isnan(transformed).any()


def test_feature_names_match_transformed_shape():
    X, y, numeric_cols, categorical_cols = get_test_data()

    preprocessor = build_preprocessor(numeric_cols, categorical_cols)

    transformed = preprocessor.fit_transform(X.iloc[:100], y.iloc[:100])
    feature_names = get_feature_names(preprocessor)

    assert len(feature_names) == transformed.shape[1]


def test_saved_and_reloaded_preprocessor_is_identical():
    X, y, numeric_cols, categorical_cols = get_test_data()

    preprocessor = build_preprocessor(numeric_cols, categorical_cols)

    X_train = X.iloc[:100]

    transformed_original = preprocessor.fit_transform(
        X_train, y.iloc[:100]
    )

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(preprocessor, MODEL_PATH)

    loaded_preprocessor = joblib.load(MODEL_PATH)

    transformed_loaded = loaded_preprocessor.transform(X_train)

    assert np.allclose(
        transformed_original,
        transformed_loaded,
    )


def test_select_top_features_returns_k_features():
    X, y, numeric_cols, categorical_cols = get_test_data()

    preprocessor = build_preprocessor(numeric_cols, categorical_cols)

    transformed = preprocessor.fit_transform(
        X.iloc[:100],
        y.iloc[:100],
    )

    feature_names = get_feature_names(preprocessor)

    transformed_df = pd.DataFrame(
        transformed,
        columns=feature_names,
    )

    selected = select_top_features(
        transformed_df,
        y.iloc[:100],
        k=5,
    )

    assert len(selected) == 5