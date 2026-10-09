
"""Tests for the final Epic 5 ML pipeline."""

import json

import joblib
import numpy as np
import pandas as pd
import pytest

from sklearn.linear_model import (
    LinearRegression,
    LogisticRegression,
)
from sklearn.pipeline import Pipeline

from epic5_advanced_ml.project_final_ml_pipeline import (
    MLPipelineRunner,
    build_leaderboard,
)


@pytest.fixture
def runner_factory(tmp_path):
    """Create a runner using the real project datasets."""

    def create_runner(task="classification", output_dir=None):
        if output_dir is None:
            output_dir = tmp_path / task

        return MLPipelineRunner(
            task=task,
            output_dir=output_dir,
        )

    return create_runner


def test_invalid_task_raises_value_error(tmp_path):
    with pytest.raises(ValueError, match="task must be"):
        MLPipelineRunner(
            task="invalid",
            output_dir=tmp_path,
        )


@pytest.mark.parametrize(
    "task",
    ["classification", "regression"],
)
def test_runner_initializes_and_splits_data(
    runner_factory,
    task,
):
    runner = runner_factory(task)

    assert len(runner.X_train) > 0
    assert len(runner.X_test) > 0
    assert len(runner.y_train) == len(runner.X_train)
    assert len(runner.y_test) == len(runner.X_test)


@pytest.mark.parametrize(
    "task",
    ["classification", "regression"],
)
def test_preprocessor_is_available(runner_factory, task):
    runner = runner_factory(task)

    assert runner.preprocessor is not None


@pytest.mark.parametrize(
    "task",
    ["classification", "regression"],
)
def test_pipeline_contains_preprocessor_and_model(
    runner_factory,
    task,
):
    runner = runner_factory(task)
    pipeline = runner._build_pipeline(
        runner._build_candidates()[
            next(iter(runner._build_candidates()))
        ]
    )

    assert isinstance(pipeline, Pipeline)
    assert "preprocessor" in pipeline.named_steps
    assert "model" in pipeline.named_steps


@pytest.mark.parametrize(
    "task",
    ["classification", "regression"],
)
def test_candidate_models_are_available(
    runner_factory,
    task,
):
    runner = runner_factory(task)
    candidates = runner._build_candidates()

    assert candidates
    assert "knn" in candidates
    assert "svm" in candidates
    assert "decision_tree" in candidates
    assert "random_forest" in candidates
    assert "gradient_boosting" in candidates
    assert "stacking" in candidates


def test_classification_candidates_include_logistic_regression(
    runner_factory,
):
    runner = runner_factory("classification")

    assert "logistic_regression" in runner._build_candidates()


def test_regression_candidates_include_linear_regression(
    runner_factory,
):
    runner = runner_factory("regression")

    assert "linear_regression" in runner._build_candidates()


def test_primary_metric_for_classification(runner_factory):
    runner = runner_factory("classification")

    assert runner._primary_metric() == "f1"


def test_primary_metric_for_regression(runner_factory):
    runner = runner_factory("regression")

    assert (
        runner._primary_metric()
        == "neg_root_mean_squared_error"
    )


def test_evaluate_classification_returns_metrics(
    runner_factory,
):
    runner = runner_factory("classification")
    model = runner._build_pipeline(
        LogisticRegression(max_iter=1000)
    )
    model.fit(runner.X_train, runner.y_train)

    metrics = runner._evaluate(model)

    assert "accuracy" in metrics
    assert "precision" in metrics
    assert "recall" in metrics
    assert "f1" in metrics
    assert 0 <= metrics["accuracy"] <= 1


def test_evaluate_regression_returns_metrics(
    runner_factory,
):
    runner = runner_factory("regression")
    model = runner._build_pipeline(LinearRegression())
    model.fit(runner.X_train, runner.y_train)

    metrics = runner._evaluate(model)

    assert "mae" in metrics
    assert "mse" in metrics
    assert "rmse" in metrics
    assert "r2" in metrics
    assert metrics["rmse"] >= 0


@pytest.mark.parametrize(
    "task",
    ["classification", "regression"],
)
def test_run_all_models_creates_leaderboard(
    runner_factory,
    task,
):
    runner = runner_factory(task)

    leaderboard = runner.run_all_models()

    assert not leaderboard.empty
    assert "model" in leaderboard.columns
    assert "cv_mean" in leaderboard.columns
    assert "cv_std" in leaderboard.columns
    assert (runner.output_dir / "leaderboard.csv").exists()


def test_run_all_models_reports_when_all_models_fail(
    runner_factory,
):
    runner = runner_factory("classification")
    runner._build_candidates = lambda: {
        "broken_model": object(),
    }

    with pytest.raises(
        RuntimeError,
        match="All model candidates failed",
    ):
        runner.run_all_models()


def test_tune_top_candidates_runs_when_leaderboard_is_empty(
    runner_factory,
):
    runner = runner_factory("classification")

    leaderboard = runner.tune_top_candidates(n=1)

    assert not leaderboard.empty


def test_tune_top_candidates_rejects_zero(
    runner_factory,
):
    runner = runner_factory("classification")

    with pytest.raises(ValueError, match="at least 1"):
        runner.tune_top_candidates(n=0)


def test_cross_validate_final_returns_results(
    runner_factory,
):
    runner = runner_factory("classification")
    runner.run_all_models()

    results = runner.cross_validate_final()

    assert "cv_mean" in results
    assert "cv_std" in results
    assert len(results["fold_scores"]) == 5


def test_cross_validate_final_rejects_unknown_model(
    runner_factory,
):
    runner = runner_factory("classification")

    with pytest.raises(ValueError, match="Model not found"):
        runner.cross_validate_final("unknown_model")


@pytest.mark.parametrize(
    "task",
    ["classification", "regression"],
)
def test_save_best_creates_output_files(
    runner_factory,
    task,
):
    runner = runner_factory(task)
    runner.run_all_models()

    model_path = runner.save_best()

    assert model_path.exists()
    assert (runner.output_dir / "metrics.json").exists()
    assert (runner.output_dir / "leaderboard.csv").exists()
    assert (
        runner.output_dir / "MODEL_SELECTION.md"
    ).exists()


def test_save_best_supports_custom_output_directory(
    runner_factory,
    tmp_path,
):
    runner = runner_factory("regression")
    runner.run_all_models()

    custom_dir = tmp_path / "custom_results"
    model_path = runner.save_best(custom_dir)

    assert model_path.parent == custom_dir
    assert (custom_dir / "metrics.json").exists()


def test_saved_metrics_are_valid_json(runner_factory):
    runner = runner_factory("classification")
    runner.run_all_models()
    runner.save_best()

    with (
        runner.output_dir / "metrics.json"
    ).open("r", encoding="utf-8") as file:
        saved_metrics = json.load(file)

    assert saved_metrics["task"] == "classification"
    assert "accuracy" in saved_metrics["test_metrics"]
    assert "best_model" in saved_metrics


def test_report_does_not_require_tabulate(runner_factory):
    runner = runner_factory("regression")
    runner.run_all_models()
    runner.save_best()

    report = (
        runner.output_dir / "MODEL_SELECTION.md"
    ).read_text(encoding="utf-8")

    assert "# Model Selection" in report
    assert "## Leaderboard" in report


def test_saved_model_can_be_reloaded_and_predict(
    runner_factory,
):
    runner = runner_factory("classification")
    runner.run_all_models()
    model_path = runner.save_best()

    loaded_model = joblib.load(model_path)
    predictions = loaded_model.predict(runner.X_test)

    assert len(predictions) == len(runner.y_test)


def test_load_best_loads_saved_model_without_training(
    runner_factory,
):
    output_dir = None
    runner = runner_factory("classification", output_dir)
    runner.run_all_models()
    runner.save_best()

    new_runner = runner_factory(
        "classification",
        runner.output_dir,
    )

    def fail_if_called():
        raise AssertionError("Models should not be retrained")

    new_runner.run_all_models = fail_if_called

    loaded_model = new_runner.load_best()

    assert loaded_model is not None
    assert new_runner.best_model_name is not None


def test_load_best_raises_when_model_file_is_missing(
    runner_factory,
):
    runner = runner_factory("classification")

    with pytest.raises(FileNotFoundError, match="Saved model not found"):
        runner.load_best()


def test_build_leaderboard_sorts_classification_descending():
    results = {
        "model_a": {
            "cv_mean": 0.80,
            "cv_std": 0.02,
        },
        "model_b": {
            "cv_mean": 0.95,
            "cv_std": 0.01,
        },
        "task": "classification",
    }

    leaderboard = build_leaderboard(results)

    assert leaderboard["model"].tolist() == [
        "model_b",
        "model_a",
    ]


def test_build_leaderboard_sorts_regression_ascending():
    results = {
        "model_a": {
            "cv_mean": 0.55,
            "cv_std": 0.02,
        },
        "model_b": {
            "cv_mean": 0.30,
            "cv_std": 0.01,
        },
        "task": "regression",
    }

    leaderboard = build_leaderboard(results)

    assert leaderboard["model"].tolist() == [
        "model_b",
        "model_a",
    ]


def test_build_leaderboard_returns_empty_for_no_results():
    assert build_leaderboard({}).empty


def test_build_leaderboard_requires_cv_mean():
    results = {
        "model_a": {
            "accuracy": 0.90,
        },
        "task": "classification",
    }

    with pytest.raises(
        ValueError,
        match="cv_mean",
    ):
        build_leaderboard(results)


def test_loaded_model_predictions_match_saved_model(
    runner_factory,
):
    runner = runner_factory("regression")
    runner.run_all_models()
    model_path = runner.save_best()

    original_model = joblib.load(model_path)
    original_predictions = original_model.predict(
        runner.X_test
    )

    new_runner = runner_factory(
        "regression",
        runner.output_dir,
    )
    loaded_model = new_runner.load_best()
    loaded_predictions = loaded_model.predict(
        new_runner.X_test
    )

    np.testing.assert_allclose(
        original_predictions,
        loaded_predictions,
    )
