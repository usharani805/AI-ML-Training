import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, r2_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor


CLASSIFICATION_DATA = Path("epic4_ml/data/breast_cancer.csv")
REGRESSION_DATA = Path("epic4_ml/data/california_housing.csv")


def save_model_bundle(
    model,
    metrics: dict,
    output_dir: str,
) -> None:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    joblib.dump(
        model,
        output_path / "model.joblib",
    )

    with open(
        output_path / "metrics.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            metrics,
            file,
            indent=4,
        )


def train_model(task: str, model_name: str):
    if task == "classification":
        df = pd.read_csv(CLASSIFICATION_DATA)

        X = df.drop(columns=["target"])
        y = df["target"]

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42,
            stratify=y,
        )

        if model_name == "decision_tree":
            model = DecisionTreeClassifier(
                random_state=42
            )
        elif model_name == "random_forest":
            model = RandomForestClassifier(
                n_estimators=100,
                random_state=42,
            )
        else:
            raise ValueError(
                "Unsupported classification model"
            )

        model.fit(X_train, y_train)

        predictions = model.predict(X_test)

        metrics = {
            "task": task,
            "model": model_name,
            "accuracy": accuracy_score(
                y_test,
                predictions,
            ),
        }

        return model, metrics

    if task == "regression":
        df = pd.read_csv(REGRESSION_DATA)

        X = df.drop(columns=["MedHouseVal"])
        y = df["MedHouseVal"]

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42,
        )

        if model_name == "decision_tree":
            model = DecisionTreeRegressor(
                random_state=42
            )
        elif model_name == "random_forest":
            model = RandomForestRegressor(
                n_estimators=100,
                random_state=42,
            )
        else:
            raise ValueError(
                "Unsupported regression model"
            )

        model.fit(X_train, y_train)

        predictions = model.predict(X_test)

        metrics = {
            "task": task,
            "model": model_name,
            "r2_score": r2_score(
                y_test,
                predictions,
            ),
        }

        return model, metrics

    raise ValueError(
        "task must be 'classification' or 'regression'"
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--task",
        required=True,
        choices=["classification", "regression"],
    )

    parser.add_argument(
        "--model",
        required=True,
        choices=["decision_tree", "random_forest"],
    )

    parser.add_argument(
        "--output-dir",
        required=True,
    )

    args = parser.parse_args()

    model, metrics = train_model(
        args.task,
        args.model,
    )

    save_model_bundle(
        model,
        metrics,
        args.output_dir,
    )

    print("Model saved successfully.")
    print("Metrics:")
    print(json.dumps(metrics, indent=4))


if __name__ == "__main__":
    main()