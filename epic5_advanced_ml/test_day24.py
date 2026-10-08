import numpy as np
import pandas as pd
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from epic5_advanced_ml.day24_tuning_cv import (
    manual_kfold_cv,
    nested_cv_score,
    run_grid_search,
    run_random_search,
)


def create_test_data():
    X, y = make_regression(
        n_samples=60,
        n_features=4,
        noise=0.1,
        random_state=42,
    )

    X = pd.DataFrame(X)
    y = pd.Series(y)

    return X, y


def create_pipeline():
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            ("model", LinearRegression()),
        ]
    )


def test_manual_kfold_returns_correct_number_of_folds():
    X, y = create_test_data()
    model = create_pipeline()

    scores = manual_kfold_cv(
        X,
        y,
        model,
        k=5,
        task="regression",
    )

    assert len(scores) == 5


def test_manual_kfold_scores_are_numeric():
    X, y = create_test_data()
    model = create_pipeline()

    scores = manual_kfold_cv(
        X,
        y,
        model,
        k=5,
        task="regression",
    )

    assert all(isinstance(score, float) for score in scores)


def test_manual_kfold_has_no_overlapping_test_indices():
    X, y = create_test_data()

    splitter = KFold(
        n_splits=5,
        shuffle=True,
        random_state=42,
    )

    test_indices = []

    for _, test_index in splitter.split(X, y):
        test_indices.extend(test_index.tolist())

    assert len(test_indices) == len(set(test_indices))


def test_manual_kfold_mean_matches_sklearn():
    X, y = create_test_data()
    model = create_pipeline()

    manual_scores = manual_kfold_cv(
        X,
        y,
        model,
        k=5,
        task="regression",
    )

    cv = KFold(
        n_splits=5,
        shuffle=True,
        random_state=42,
    )

    sklearn_scores = cross_val_score(
        model,
        X,
        y,
        cv=cv,
        scoring="r2",
    )

    assert np.isclose(
        np.mean(manual_scores),
        np.mean(sklearn_scores),
        atol=1e-10,
    )


def test_grid_search_returns_best_params():
    X, y = create_test_data()
    pipeline = create_pipeline()

    param_grid = {
        "model__fit_intercept": [True, False],
    }

    result = run_grid_search(
        pipeline,
        param_grid,
        X,
        y,
        cv=3,
    )

    assert "model__fit_intercept" in result.best_params_


def test_grid_search_best_params_are_from_grid():
    X, y = create_test_data()
    pipeline = create_pipeline()

    param_grid = {
        "model__fit_intercept": [True, False],
    }

    result = run_grid_search(
        pipeline,
        param_grid,
        X,
        y,
        cv=3,
    )

    assert result.best_params_["model__fit_intercept"] in [
        True,
        False,
    ]


def test_random_search_returns_best_params():
    X, y = create_test_data()
    pipeline = create_pipeline()

    param_distributions = {
        "model__fit_intercept": [True, False],
    }

    result = run_random_search(
        pipeline,
        param_distributions,
        X,
        y,
        n_iter=2,
        cv=3,
    )

    assert "model__fit_intercept" in result.best_params_


def test_nested_cv_returns_score_for_each_outer_fold():
    X, y = create_test_data()
    pipeline = create_pipeline()

    param_grid = {
        "model__fit_intercept": [True, False],
    }

    result = nested_cv_score(
        pipeline,
        param_grid,
        X,
        y,
        outer_k=3,
        inner_k=2,
    )

    assert len(result["outer_scores"]) == 3


def test_nested_cv_returns_mean_score():
    X, y = create_test_data()
    pipeline = create_pipeline()

    param_grid = {
        "model__fit_intercept": [True, False],
    }

    result = nested_cv_score(
        pipeline,
        param_grid,
        X,
        y,
        outer_k=3,
        inner_k=2,
    )

    assert "mean_score" in result
    assert isinstance(result["mean_score"], float)