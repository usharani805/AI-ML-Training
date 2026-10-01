from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


DATA_PATH = Path("epic4_ml/data/breast_cancer.csv")
CHARTS_DIR = Path("epic4_ml/charts")


def sigmoid(z: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid function."""

    z = np.asarray(z, dtype=float)
    result = np.empty_like(z)

    positive = z >= 0
    result[positive] = 1 / (1 + np.exp(-z[positive]))

    exp_z = np.exp(z[~positive])
    result[~positive] = exp_z / (1 + exp_z)

    return result


def logistic_regression_fit(
    X,
    y,
    lr: float,
    epochs: int,
) -> tuple:
    """Fit logistic regression from scratch using gradient descent."""

    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)

    X_with_bias = np.column_stack([np.ones(X.shape[0]), X])
    weights = np.zeros(X_with_bias.shape[1])

    for _ in range(epochs):
        probabilities = sigmoid(X_with_bias @ weights)
        gradient = (
            X_with_bias.T @ (probabilities - y)
        ) / len(y)
        weights -= lr * gradient

    return weights


def classification_report_dict(
    y_true,
    y_pred,
    y_prob,
) -> dict:
    """Return binary classification metrics."""

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(
            precision_score(y_true, y_pred, zero_division=0)
        ),
        "recall": float(
            recall_score(y_true, y_pred, zero_division=0)
        ),
        "f1": float(
            f1_score(y_true, y_pred, zero_division=0)
        ),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }


def threshold_analysis(
    y_true,
    y_prob,
    thresholds: list,
) -> pd.DataFrame:
    """Compare precision, recall and F1 at different thresholds."""

    rows = []

    for threshold in thresholds:
        y_pred = (np.asarray(y_prob) >= threshold).astype(int)

        rows.append(
            {
                "threshold": threshold,
                "precision": precision_score(
                    y_true,
                    y_pred,
                    zero_division=0,
                ),
                "recall": recall_score(
                    y_true,
                    y_pred,
                    zero_division=0,
                ),
                "F1": f1_score(
                    y_true,
                    y_pred,
                    zero_division=0,
                ),
            }
        )

    return pd.DataFrame(rows)


def main():
    df = pd.read_csv(DATA_PATH)

    X = df.drop(columns=["target"])
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    preprocessor = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
        ]
    )

    logistic_pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", LogisticRegression(max_iter=1000)),
        ]
    )

    balanced_pipeline = Pipeline(
        steps=[
            ("preprocessor", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    max_iter=1000,
                    class_weight="balanced",
                ),
            ),
        ]
    )

    logistic_pipeline.fit(X_train, y_train)
    balanced_pipeline.fit(X_train, y_train)

    logistic_prob = logistic_pipeline.predict_proba(X_test)[:, 1]
    balanced_prob = balanced_pipeline.predict_proba(X_test)[:, 1]

    logistic_pred = (logistic_prob >= 0.5).astype(int)
    balanced_pred = (balanced_prob >= 0.5).astype(int)

    baseline_pred = np.full(len(y_test), y_train.mode()[0])
    baseline_prob = np.full(
        len(y_test),
        y_train.mean(),
        dtype=float,
    )

    logistic_metrics = classification_report_dict(
        y_test,
        logistic_pred,
        logistic_prob,
    )

    balanced_metrics = classification_report_dict(
        y_test,
        balanced_pred,
        balanced_prob,
    )

    baseline_metrics = classification_report_dict(
        y_test,
        baseline_pred,
        baseline_prob,
    )

    comparison = pd.DataFrame(
        [
            {
                "model": "Baseline",
                "accuracy": baseline_metrics["accuracy"],
                "precision": baseline_metrics["precision"],
                "recall": baseline_metrics["recall"],
                "F1": baseline_metrics["f1"],
                "ROC-AUC": baseline_metrics["roc_auc"],
            },
            {
                "model": "Logistic Regression",
                "accuracy": logistic_metrics["accuracy"],
                "precision": logistic_metrics["precision"],
                "recall": logistic_metrics["recall"],
                "F1": logistic_metrics["f1"],
                "ROC-AUC": logistic_metrics["roc_auc"],
            },
            {
                "model": "Balanced Logistic Regression",
                "accuracy": balanced_metrics["accuracy"],
                "precision": balanced_metrics["precision"],
                "recall": balanced_metrics["recall"],
                "F1": balanced_metrics["f1"],
                "ROC-AUC": balanced_metrics["roc_auc"],
            },
        ]
    )

    thresholds = [0.3, 0.5, 0.7]

    threshold_table = threshold_analysis(
        y_test,
        logistic_prob,
        thresholds,
    )

    print("\nClassification Metrics:")
    print("Logistic Regression:")
    print(logistic_metrics)

    print("\nBalanced Logistic Regression:")
    print(balanced_metrics)

    print("\nThreshold Analysis:")
    print(threshold_table)

    print(
        "\nChurn-prevention threshold comment:"
        "\nA lower threshold can increase recall and identify more possible churn cases."
        "\nThis may be useful when missing a potential churn customer is costly."
        "\nThe selected threshold should balance recall, precision and available retention resources."
    )

    print("\nModel Comparison:")
    print(comparison)

    CHARTS_DIR.mkdir(parents=True, exist_ok=True)

    sns.set_theme()

    fpr, tpr, _ = roc_curve(y_test, logistic_prob)

    plt.figure(figsize=(8, 6))
    plt.plot(
        fpr,
        tpr,
        label=f"Logistic Regression (AUC={logistic_metrics['roc_auc']:.3f})",
    )
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        CHARTS_DIR / "day19_roc_curve.png",
        dpi=150,
    )
    plt.close()

    cm = confusion_matrix(y_test, logistic_pred)

    plt.figure(figsize=(7, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
    )
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(
        CHARTS_DIR / "day19_confusion_matrix.png",
        dpi=150,
    )
    plt.close()

    X_train_scaled = StandardScaler().fit_transform(X_train)
    X_test_scaled = StandardScaler().fit(X_train).transform(X_test)

    weights = logistic_regression_fit(
        X_train_scaled,
        y_train,
        lr=0.1,
        epochs=5000,
    )

    from_scratch_prob = sigmoid(
        np.column_stack(
            [
                np.ones(X_test_scaled.shape[0]),
                X_test_scaled,
            ]
        )
        @ weights
    )

    from_scratch_pred = (
        from_scratch_prob >= 0.5
    ).astype(int)

    agreement = np.mean(
        from_scratch_pred == logistic_pred
    )

    print(
        f"\nFrom-scratch vs sklearn prediction agreement: "
        f"{agreement:.2%}"
    )


if __name__ == "__main__":
    main()