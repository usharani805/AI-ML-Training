import pandas as pd
import pytest

from epic4_ml.day16_ml_basics import (
    load_ml_dataset,
    split_data,
    get_baseline_scores,
    demonstrate_leakage,
)


def test_california_dataset_loads():
    X, y = load_ml_dataset("california_housing")
    assert X.shape == (20640, 8)
    assert len(y) == 20640


def test_breast_cancer_dataset_loads():
    X, y = load_ml_dataset("breast_cancer")
    assert X.shape == (569, 30)
    assert len(y) == 569


def test_split_size_is_80_20():
    X, y = load_ml_dataset("california_housing")

    X_train, X_test, y_train, y_test = split_data(
        X, y, test_size=0.2, stratify=False
    )

    assert len(X_train) == 16512
    assert len(X_test) == 4128
    assert len(y_train) == 16512
    assert len(y_test) == 4128


def test_stratified_class_ratio():
    X, y = load_ml_dataset("breast_cancer")

    _, _, y_train, y_test = split_data(
        X, y, test_size=0.2, stratify=True
    )

    original_ratio = y.value_counts(normalize=True).sort_index()
    train_ratio = y_train.value_counts(normalize=True).sort_index()
    test_ratio = y_test.value_counts(normalize=True).sort_index()

    assert (abs(original_ratio - train_ratio) <= 0.02).all()
    assert (abs(original_ratio - test_ratio) <= 0.02).all()


def test_invalid_test_size_raises_value_error():
    X, y = load_ml_dataset("california_housing")

    with pytest.raises(ValueError):
        split_data(X, y, test_size=1.0, stratify=False)


def test_empty_data_raises_value_error():
    X = pd.DataFrame()
    y = pd.Series(dtype=float)

    with pytest.raises(ValueError):
        split_data(X, y, test_size=0.2, stratify=False)


def test_baseline_function_expected_keys():
    X, y = load_ml_dataset("california_housing")

    X_train, X_test, y_train, y_test = split_data(
        X, y, test_size=0.2, stratify=False
    )

    scores = get_baseline_scores(
        X_train,
        X_test,
        y_train,
        y_test,
        "regression",
    )

    assert "baseline_score" in scores


def test_leakage_demo_returns_expected_keys():
    X, y = load_ml_dataset("california_housing")

    result = demonstrate_leakage(X, y)

    assert "leakage_train_mean" in result
    assert "leakage_test_mean" in result
    assert "no_leakage_train_mean" in result
    assert "no_leakage_test_mean" in result