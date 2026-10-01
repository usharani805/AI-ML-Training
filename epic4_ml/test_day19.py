import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from epic4_ml.day19_logistic_regression import (
    classification_report_dict,
    logistic_regression_fit,
    sigmoid,
    threshold_analysis,
)


def test_sigmoid_zero():
    result = sigmoid(np.array([0.0]))
    assert result[0] == 0.5


def test_sigmoid_large_values_stable():
    result = sigmoid(np.array([-1000.0, 1000.0]))
    assert np.all(np.isfinite(result))
    assert result[0] < 0.001
    assert result[1] > 0.999


def test_sigmoid_output_range():
    values = sigmoid(np.array([-10.0, -1.0, 0.0, 1.0, 10.0]))
    assert np.all(values >= 0)
    assert np.all(values <= 1)


def test_logistic_regression_fit_shape():
    X = np.array(
        [
            [0.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [1.0, 1.0],
        ]
    )
    y = np.array([0, 0, 1, 1])

    weights = logistic_regression_fit(
        X,
        y,
        lr=0.1,
        epochs=100,
    )

    assert weights.shape == (3,)


def test_from_scratch_agrees_with_sklearn():
    df = pd.read_csv("epic4_ml/data/breast_cancer.csv")

    X = df.drop(columns=["target"])
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    weights = logistic_regression_fit(
        X_train_scaled,
        y_train,
        lr=0.1,
        epochs=5000,
    )

    scratch_prob = sigmoid(
        np.column_stack(
            [
                np.ones(X_test_scaled.shape[0]),
                X_test_scaled,
            ]
        )
        @ weights
    )

    scratch_pred = (scratch_prob >= 0.5).astype(int)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_scaled, y_train)

    sklearn_pred = model.predict(X_test_scaled)

    agreement = np.mean(scratch_pred == sklearn_pred)

    assert agreement >= 0.90


def test_classification_metrics_in_range():
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 1, 1, 1])
    y_prob = np.array([0.1, 0.6, 0.8, 0.9])

    metrics = classification_report_dict(
        y_true,
        y_pred,
        y_prob,
    )

    for metric in [
        "accuracy",
        "precision",
        "recall",
        "f1",
        "roc_auc",
    ]:
        assert 0 <= metrics[metric] <= 1


def test_threshold_analysis_row_count():
    y_true = np.array([0, 0, 1, 1])
    y_prob = np.array([0.1, 0.4, 0.6, 0.9])
    thresholds = [0.3, 0.5, 0.7]

    result = threshold_analysis(
        y_true,
        y_prob,
        thresholds,
    )

    assert len(result) == len(thresholds)


def test_threshold_analysis_columns_and_values():
    y_true = np.array([0, 0, 1, 1])
    y_prob = np.array([0.1, 0.4, 0.6, 0.9])
    thresholds = [0.3, 0.5, 0.7]

    result = threshold_analysis(
        y_true,
        y_prob,
        thresholds,
    )

    assert list(result.columns) == [
        "threshold",
        "precision",
        "recall",
        "F1",
    ]

    assert result["precision"].between(0, 1).all()
    assert result["recall"].between(0, 1).all()
    assert result["F1"].between(0, 1).all()