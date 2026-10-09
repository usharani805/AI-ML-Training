"""Tests for the Epic 5 final ML pipeline."""

import json

import joblib
import pandas as pd
import pytest
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.pipeline import Pipeline

from common.ml_utils import (
    build_day17_preprocessor,
    load_dataset,
)
from epic5_advanced_ml.project_final_ml_pipeline import (
    MLPipelineRunner,
)


def use_one_model(runner, model_name, estimator):
    """Use one lightweight candidate for focused tests."""
    runner.models = {
        model_name: Pipeline(
            steps=[
                ("preprocessor", runner.preprocessor),
                ("model", estimator),
            ]
        )
    }
    runner._build_candidates = lambda: runner.models


def test_load_classification_dataset():
    X, y = load_dataset("classification")
    assert len(X) == 569
    assert len(y) == 569


def test_load_regression_dataset():
    X, y = load_dataset("regression")
    assert len(X) == 20640
    assert len(y) == 20640


def test_invalid_dataset_task():
    with pytest.raises(ValueError):
        load_dataset("invalid")


def test_day17_preprocessor():
    X, _ = load_dataset("regression")
    preprocessor = build_day17_preprocessor(X)
    assert isinstance(preprocessor, ColumnTransformer)


def test_invalid_runner_task():
    with pytest.raises(ValueError):
        MLPipelineRunner("invalid")


def test_classification_candidates():
    runner = MLPipelineRunner("classification")
    candidates = runner._build_candidates()
    assert "logistic_regression" in candidates
    assert "random_forest" in candidates


def test_regression_candidates():
    runner = MLPipelineRunner("regression")
    candidates = runner._build_candidates()
    assert "linear_regression" in candidates
    assert "random_forest" in candidates


def test_classification_leaderboard_sorted(tmp_path):
    runner = MLPipelineRunner(
        "classification",
        str(tmp_path),
    )
    use_one_model(
        runner,
        "logistic_regression",
        LogisticRegression(max_iter=2000),
    )

    leaderboard = runner.run_all_models()

    assert not leaderboard.empty
    assert leaderboard["cv_mean"].is_monotonic_decreasing


def test_regression_leaderboard_sorted(tmp_path):
    runner = MLPipelineRunner(
        "regression",
        str(tmp_path),
    )
    use_one_model(
        runner,
        "linear_regression",
        LinearRegression(),
    )

    leaderboard = runner.run_all_models()

    assert not leaderboard.empty
    assert leaderboard["cv_mean"].is_monotonic_increasing


def test_failed_model_is_skipped(tmp_path, caplog):
    runner = MLPipelineRunner(
        "classification",
        str(tmp_path),
    )
    runner.models = {
        "broken_model": Pipeline(
            steps=[
                ("preprocessor", runner.preprocessor),
                ("model", LogisticRegression(max_iter=-1)),
            ]
        ),
        "working_model": Pipeline(
            steps=[
                ("preprocessor", runner.preprocessor),
                ("model", LogisticRegression(max_iter=2000)),
            ]
        ),
    }
    runner._build_candidates = lambda: runner.models

    leaderboard = runner.run_all_models()

    assert "working_model" in leaderboard["model"].tolist()
    assert "broken_model" not in leaderboard["model"].tolist()
    assert "Skipping failed model" in caplog.text


def test_cross_validation_result(tmp_path):
    runner = MLPipelineRunner(
        "classification",
        str(tmp_path),
    )
    model = Pipeline(
        steps=[
            ("preprocessor", runner.preprocessor),
            ("model", LogisticRegression(max_iter=2000)),
        ]
    )

    result = runner.cross_validate_final(model)

    assert "cv_mean" in result
    assert "cv_std" in result
    assert result["cv_std"] >= 0


def test_save_best_creates_files(tmp_path):
    runner = MLPipelineRunner(
        "classification",
        str(tmp_path),
    )
    use_one_model(
        runner,
        "logistic_regression",
        LogisticRegression(max_iter=2000),
    )
    runner.run_all_models()

    output_dir = tmp_path / "saved"
    runner.save_best(str(output_dir))

    assert (output_dir / "model.joblib").exists()
    assert (output_dir / "metrics.json").exists()
    assert (output_dir / "leaderboard.csv").exists()
    assert (output_dir / "MODEL_SELECTION.md").exists()


def test_saved_model_can_predict(tmp_path):
    runner = MLPipelineRunner(
        "classification",
        str(tmp_path),
    )
    use_one_model(
        runner,
        "logistic_regression",
        LogisticRegression(max_iter=2000),
    )
    runner.run_all_models()

    output_dir = tmp_path / "saved"
    runner.save_best(str(output_dir))

    saved_model = joblib.load(output_dir / "model.joblib")
    predictions = saved_model.predict(runner.X_test)

    assert len(predictions) == len(runner.y_test)


def test_saved_metrics_are_valid_json(tmp_path):
    runner = MLPipelineRunner(
        "classification",
        str(tmp_path),
    )
    use_one_model(
        runner,
        "logistic_regression",
        LogisticRegression(max_iter=2000),
    )
    runner.run_all_models()

    output_dir = tmp_path / "saved"
    runner.save_best(str(output_dir))

    with open(
        output_dir / "metrics.json",
        encoding="utf-8",
    ) as file:
        metrics = json.load(file)

    assert metrics["task"] == "classification"
    assert "accuracy" in metrics
    assert "f1" in metrics