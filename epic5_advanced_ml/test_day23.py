from pathlib import Path

import pytest

from epic5_advanced_ml.day23_ensembles import (
    build_stacking_model,
    compare_feature_importance,
    load_classification_data,
    load_regression_data,
    split_classification_data,
    split_regression_data,
    split_validation_data,
    train_bagging_models,
    train_boosting_models,
    train_gradient_boosting_model,
)

from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import accuracy_score, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor


def test_classification_data_shape():
    X, y = load_classification_data()

    assert X.shape == (569, 30)
    assert y.shape == (569,)


def test_regression_data_shape():
    X, y = load_regression_data()

    assert X.shape == (20640, 8)
    assert y.shape == (20640,)


def test_bagging_classification_models():
    X, y = load_classification_data()

    X_train, X_test, y_train, y_test = (
        split_classification_data(X, y)
    )

    models = train_bagging_models(
        X_train,
        y_train,
        "classification",
    )

    for model in models.values():
        predictions = model.predict(X_test)

        assert len(predictions) == len(y_test)


def test_bagging_regression_models():
    X, y = load_regression_data()

    X_train, X_test, y_train, y_test = (
        split_regression_data(X, y)
    )

    models = train_bagging_models(
        X_train,
        y_train,
        "regression",
    )

    for model in models.values():
        predictions = model.predict(X_test)

        assert len(predictions) == len(y_test)


def test_gradient_boosting_classification():
    X, y = load_classification_data()

    X_train, X_test, y_train, y_test = (
        split_classification_data(X, y)
    )

    model = train_gradient_boosting_model(
        X_train,
        y_train,
        "classification",
    )

    predictions = model.predict(X_test)

    assert len(predictions) == len(y_test)


def test_gradient_boosting_regression():
    X, y = load_regression_data()

    X_train, X_test, y_train, y_test = (
        split_regression_data(X, y)
    )

    model = train_gradient_boosting_model(
        X_train,
        y_train,
        "regression",
    )

    predictions = model.predict(X_test)

    assert len(predictions) == len(y_test)


def test_xgboost_lightgbm_classification_early_stopping():
    X, y = load_classification_data()

    X_train, X_test, y_train, y_test = (
        split_classification_data(X, y)
    )

    X_train, X_val, y_train, y_val = (
        split_validation_data(
            X_train,
            y_train,
            "classification",
        )
    )

    models, histories = train_boosting_models(
        X_train,
        y_train,
        X_val,
        y_val,
        "classification",
    )

    assert "XGBoost" in models
    assert "LightGBM" in models
    assert "XGBoost" in histories
    assert "LightGBM" in histories

    assert models["XGBoost"].best_iteration < 300
    assert models["LightGBM"].best_iteration_ < 300


def test_xgboost_lightgbm_regression_early_stopping():
    X, y = load_regression_data()

    X_train, X_test, y_train, y_test = (
        split_regression_data(X, y)
    )

    X_train, X_val, y_train, y_val = (
        split_validation_data(
            X_train,
            y_train,
            "regression",
        )
    )

    models, histories = train_boosting_models(
        X_train,
        y_train,
        X_val,
        y_val,
        "regression",
    )

    assert "XGBoost" in models
    assert "LightGBM" in models
    assert "XGBoost" in histories
    assert "LightGBM" in histories

    assert models["XGBoost"].best_iteration < 300
    assert models["LightGBM"].best_iteration_ < 300


def test_stacking_classification():
    X, y = load_classification_data()

    X_train, X_test, y_train, y_test = (
        split_classification_data(X, y)
    )

    base_models = [
        (
            "decision_tree",
            DecisionTreeClassifier(
                max_depth=5,
                random_state=42,
            ),
        ),
        (
            "svm",
            make_pipeline(
                StandardScaler(),
                SVC(
                    probability=True,
                    random_state=42,
                ),
            ),
        ),
        (
            "logistic",
            make_pipeline(
                StandardScaler(),
                LogisticRegression(
                    max_iter=1000,
                    random_state=42,
                ),
            ),
        ),
    ]

    model = build_stacking_model(
        base_models,
        LogisticRegression(
            max_iter=1000,
            random_state=42,
        ),
        "classification",
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    assert len(predictions) == len(y_test)


def test_stacking_regression():
    X, y = load_regression_data()

    X_train, X_test, y_train, y_test = (
        split_regression_data(X, y)
    )

    base_models = [
        (
            "decision_tree",
            DecisionTreeRegressor(
                max_depth=5,
                random_state=42,
            ),
        ),
        (
            "svm",
            make_pipeline(
                StandardScaler(),
                SVR(),
            ),
        ),
        (
            "linear",
            make_pipeline(
                StandardScaler(),
                LinearRegression(),
            ),
        ),
    ]

    model = build_stacking_model(
        base_models,
        LinearRegression(),
        "regression",
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    assert len(predictions) == len(y_test)


def test_feature_importance_comparison():
    X, y = load_regression_data()

    X_train, X_test, y_train, y_test = (
        split_regression_data(X, y)
    )

    models = train_bagging_models(
        X_train,
        y_train,
        "regression",
    )

    boosting_models, _ = train_boosting_models(
        X_train,
        y_train,
        X_test,
        y_test,
        "regression",
    )

    models.update(boosting_models)

    importance = compare_feature_importance(
        models,
        X.columns.tolist(),
    )

    assert not importance.empty
    assert len(importance.index) == len(X.columns)