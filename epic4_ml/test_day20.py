import json
import subprocess
import sys

import matplotlib

matplotlib.use("Agg")

import joblib
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from epic4_ml.day20_trees_and_project import (
    depth_vs_score_analysis,
    main,
    plot_feature_importance,
    plot_small_tree,
    train_and_compare_models,
)
from epic4_ml.train import save_model_bundle, train_model


CLASSIFICATION_DATA = "epic4_ml/data/breast_cancer.csv"
REGRESSION_DATA = "epic4_ml/data/california_housing.csv"


def get_classification_data():
    df = pd.read_csv(CLASSIFICATION_DATA)
    X = df.drop(columns=["target"])
    y = df["target"]

    return train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )


def get_regression_data():
    df = pd.read_csv(REGRESSION_DATA)
    X = df.drop(columns=["MedHouseVal"])
    y = df["MedHouseVal"]

    return train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
    )


def test_classification_depth_analysis():
    X_train, X_test, y_train, y_test = get_classification_data()

    result = depth_vs_score_analysis(
        X_train,
        X_test,
        y_train,
        y_test,
        [2, 4, 6, 10, None],
    )

    assert len(result) == 5


def test_depth_score_columns():
    X_train, X_test, y_train, y_test = get_classification_data()

    result = depth_vs_score_analysis(
        X_train,
        X_test,
        y_train,
        y_test,
        [2, 4, 6, 10, None],
    )

    assert list(result.columns) == [
        "max_depth",
        "train_score",
        "test_score",
    ]


def test_depth_scores_are_valid():
    X_train, X_test, y_train, y_test = get_classification_data()

    result = depth_vs_score_analysis(
        X_train,
        X_test,
        y_train,
        y_test,
        [2, 4, 6, 10, None],
    )

    assert result["train_score"].between(0, 1).all()
    assert result["test_score"].between(0, 1).all()


def test_regression_depth_analysis():
    X_train, X_test, y_train, y_test = get_regression_data()

    result = depth_vs_score_analysis(
        X_train,
        X_test,
        y_train,
        y_test,
        [2, 4, 6, 10, None],
    )

    assert len(result) == 5


def test_regression_depth_columns():
    X_train, X_test, y_train, y_test = get_regression_data()

    result = depth_vs_score_analysis(
        X_train,
        X_test,
        y_train,
        y_test,
        [2, 4, 6, 10, None],
    )

    assert list(result.columns) == [
        "max_depth",
        "train_score",
        "test_score",
    ]


def test_feature_importances_sum_to_one():
    X_train, X_test, y_train, y_test = get_classification_data()

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
    )

    model.fit(X_train, y_train)

    assert sum(model.feature_importances_) == pytest.approx(1.0)


def test_feature_importance_plot(tmp_path):
    X_train, X_test, y_train, y_test = get_classification_data()

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
    )

    model.fit(X_train, y_train)

    output_path = tmp_path / "feature_importance.png"

    plot_feature_importance(
        model,
        list(X_train.columns),
        str(output_path),
    )

    assert output_path.exists()
    assert output_path.stat().st_size > 0


def test_small_tree_plot(tmp_path):
    X_train, X_test, y_train, y_test = get_classification_data()

    output_path = tmp_path / "small_tree.png"

    plot_small_tree(
        X_train,
        y_train,
        str(output_path),
    )

    assert output_path.exists()
    assert output_path.stat().st_size > 0


def test_classification_comparison_length():
    result = train_and_compare_models("classification")

    assert len(result) == 4


def test_regression_comparison_length():
    result = train_and_compare_models("regression")

    assert len(result) == 4


def test_classification_model_names():
    result = train_and_compare_models("classification")

    expected = {
        "Baseline",
        "Logistic Regression",
        "Decision Tree",
        "Random Forest",
    }

    assert set(result["model"]) == expected


def test_regression_model_names():
    result = train_and_compare_models("regression")

    expected = {
        "Baseline",
        "Linear Regression",
        "Decision Tree",
        "Random Forest",
    }

    assert set(result["model"]) == expected


def test_comparison_columns():
    result = train_and_compare_models("classification")

    assert list(result.columns) == [
        "model",
        "train_score",
        "test_score",
    ]


def test_invalid_comparison_task():
    with pytest.raises(ValueError):
        train_and_compare_models("invalid")


def test_classification_cli(tmp_path):
    output_dir = tmp_path / "classification"

    result = subprocess.run(
        [
            sys.executable,
            "epic4_ml/train.py",
            "--task",
            "classification",
            "--model",
            "random_forest",
            "--output-dir",
            str(output_dir),
        ],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert (output_dir / "model.joblib").exists()
    assert (output_dir / "metrics.json").exists()


def test_classification_cli_metrics(tmp_path):
    output_dir = tmp_path / "classification"

    subprocess.run(
        [
            sys.executable,
            "epic4_ml/train.py",
            "--task",
            "classification",
            "--model",
            "random_forest",
            "--output-dir",
            str(output_dir),
        ],
        check=True,
    )

    with open(
        output_dir / "metrics.json",
        encoding="utf-8",
    ) as file:
        metrics = json.load(file)

    assert metrics["task"] == "classification"
    assert metrics["model"] == "random_forest"
    assert "accuracy" in metrics


def test_classification_decision_tree_train():
    model, metrics = train_model(
        "classification",
        "decision_tree",
    )

    assert model is not None
    assert metrics["task"] == "classification"
    assert metrics["model"] == "decision_tree"
    assert "accuracy" in metrics


def test_classification_random_forest_train():
    model, metrics = train_model(
        "classification",
        "random_forest",
    )

    assert model is not None
    assert metrics["task"] == "classification"
    assert metrics["model"] == "random_forest"
    assert "accuracy" in metrics


def test_regression_decision_tree_train():
    model, metrics = train_model(
        "regression",
        "decision_tree",
    )

    assert model is not None
    assert metrics["task"] == "regression"
    assert metrics["model"] == "decision_tree"
    assert "r2_score" in metrics


def test_regression_random_forest_train():
    model, metrics = train_model(
        "regression",
        "random_forest",
    )

    assert model is not None
    assert metrics["task"] == "regression"
    assert metrics["model"] == "random_forest"
    assert "r2_score" in metrics


def test_invalid_classification_model():
    with pytest.raises(ValueError):
        train_model(
            "classification",
            "invalid",
        )


def test_invalid_regression_model():
    with pytest.raises(ValueError):
        train_model(
            "regression",
            "invalid",
        )


def test_invalid_train_task():
    with pytest.raises(ValueError):
        train_model(
            "invalid",
            "random_forest",
        )


def test_save_model_bundle(tmp_path):
    model, metrics = train_model(
        "classification",
        "random_forest",
    )

    save_model_bundle(
        model,
        metrics,
        str(tmp_path),
    )

    assert (tmp_path / "model.joblib").exists()
    assert (tmp_path / "metrics.json").exists()


def test_cli_regression_decision_tree(tmp_path):
    output_dir = tmp_path / "regression_tree"

    result = subprocess.run(
        [
            sys.executable,
            "epic4_ml/train.py",
            "--task",
            "regression",
            "--model",
            "decision_tree",
            "--output-dir",
            str(output_dir),
        ],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert (output_dir / "model.joblib").exists()
    assert (output_dir / "metrics.json").exists()


def test_cli_classification_decision_tree(tmp_path):
    output_dir = tmp_path / "classification_tree"

    result = subprocess.run(
        [
            sys.executable,
            "epic4_ml/train.py",
            "--task",
            "classification",
            "--model",
            "decision_tree",
            "--output-dir",
            str(output_dir),
        ],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert (output_dir / "model.joblib").exists()
    assert (output_dir / "metrics.json").exists()


def test_cli_regression_random_forest(tmp_path):
    output_dir = tmp_path / "regression_forest"

    result = subprocess.run(
        [
            sys.executable,
            "epic4_ml/train.py",
            "--task",
            "regression",
            "--model",
            "random_forest",
            "--output-dir",
            str(output_dir),
        ],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert (output_dir / "model.joblib").exists()
    assert (output_dir / "metrics.json").exists()


def test_main_creates_charts():
    main()

    assert True


def test_saved_model_reloads_and_predicts(tmp_path):
    model, metrics = train_model(
        "classification",
        "random_forest",
    )

    save_model_bundle(
        model,
        metrics,
        str(tmp_path),
    )

    loaded_model = joblib.load(
        tmp_path / "model.joblib"
    )

    df = pd.read_csv(CLASSIFICATION_DATA)

    X = df.drop(columns=["target"])

    original_predictions = model.predict(X)
    loaded_predictions = loaded_model.predict(X)

    assert loaded_predictions.shape == original_predictions.shape