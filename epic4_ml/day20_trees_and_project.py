from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, plot_tree


CLASSIFICATION_DATA = Path("epic4_ml/data/breast_cancer.csv")
REGRESSION_DATA = Path("epic4_ml/data/california_housing.csv")
CHARTS_DIR = Path("epic4_ml/charts")


def load_classification_data():
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


def load_regression_data():
    df = pd.read_csv(REGRESSION_DATA)

    X = df.drop(columns=["MedHouseVal"])
    y = df["MedHouseVal"]

    return train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
    )


def depth_vs_score_analysis(
    X_train,
    X_test,
    y_train,
    y_test,
    depths: list,
) -> pd.DataFrame:
    results = []

    classification = y_train.nunique() <= 2

    for depth in depths:
        if classification:
            model = DecisionTreeClassifier(
                max_depth=depth,
                random_state=42,
            )
        else:
            model = DecisionTreeRegressor(
                max_depth=depth,
                random_state=42,
            )

        model.fit(X_train, y_train)

        results.append(
            {
                "max_depth": depth,
                "train_score": model.score(X_train, y_train),
                "test_score": model.score(X_test, y_test),
            }
        )

    return pd.DataFrame(results)


def plot_small_tree(
    X,
    y,
    save_path,
) -> None:
    model = DecisionTreeClassifier(
        max_depth=3,
        random_state=42,
    )

    model.fit(X, y)

    plt.figure(figsize=(16, 10))

    plot_tree(
        model,
        feature_names=list(X.columns),
        class_names=[str(value) for value in model.classes_],
        filled=True,
        rounded=True,
    )

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_feature_importance(
    model,
    feature_names: list,
    save_path: str,
) -> None:
    importance = pd.Series(
        model.feature_importances_,
        index=feature_names,
    ).sort_values(ascending=False).head(10)

    plt.figure(figsize=(10, 6))

    importance.sort_values().plot(kind="barh")

    plt.title("Top 10 Feature Importances")
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def train_and_compare_models(task: str) -> pd.DataFrame:
    if task == "classification":
        X_train, X_test, y_train, y_test = load_classification_data()

        models = {
            "Baseline": Pipeline(
                [
                    (
                        "model",
                        DummyClassifier(
                            strategy="most_frequent"
                        ),
                    )
                ]
            ),
            "Logistic Regression": Pipeline(
                [
                    (
                        "model",
                        LogisticRegression(
                            max_iter=1000,
                            random_state=42,
                        ),
                    )
                ]
            ),
            "Decision Tree": Pipeline(
                [
                    (
                        "model",
                        DecisionTreeClassifier(
                            random_state=42,
                        ),
                    )
                ]
            ),
            "Random Forest": Pipeline(
                [
                    (
                        "model",
                        RandomForestClassifier(
                            n_estimators=100,
                            random_state=42,
                        ),
                    )
                ]
            ),
        }

    elif task == "regression":
        X_train, X_test, y_train, y_test = load_regression_data()

        models = {
            "Baseline": Pipeline(
                [
                    (
                        "model",
                        DummyRegressor(
                            strategy="mean"
                        ),
                    )
                ]
            ),
            "Linear Regression": Pipeline(
                [
                    (
                        "model",
                        LinearRegression(),
                    )
                ]
            ),
            "Decision Tree": Pipeline(
                [
                    (
                        "model",
                        DecisionTreeRegressor(
                            random_state=42,
                        ),
                    )
                ]
            ),
            "Random Forest": Pipeline(
                [
                    (
                        "model",
                        RandomForestRegressor(
                            n_estimators=100,
                            random_state=42,
                        ),
                    )
                ]
            ),
        }

    else:
        raise ValueError(
            "task must be 'classification' or 'regression'"
        )

    results = []

    for name, model in models.items():
        model.fit(X_train, y_train)

        results.append(
            {
                "model": name,
                "train_score": model.score(
                    X_train,
                    y_train,
                ),
                "test_score": model.score(
                    X_test,
                    y_test,
                ),
            }
        )

    return pd.DataFrame(results)


def main():
    CHARTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    X_train, X_test, y_train, y_test = load_classification_data()

    depths = [2, 4, 6, 10, None]

    depth_results = depth_vs_score_analysis(
        X_train,
        X_test,
        y_train,
        y_test,
        depths,
    )

    print("\nDecision Tree Depth Analysis:")
    print(depth_results)

    depth_labels = [
        "None" if pd.isna(value) else str(int(value))
        for value in depth_results["max_depth"]
    ]

    plt.figure(figsize=(10, 6))

    plt.plot(
        depth_labels,
        depth_results["train_score"],
        marker="o",
        label="Train Score",
    )

    plt.plot(
        depth_labels,
        depth_results["test_score"],
        marker="o",
        label="Test Score",
    )

    plt.title("Decision Tree Depth vs Score")
    plt.xlabel("Max Depth")
    plt.ylabel("Score")
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        CHARTS_DIR / "day20_depth_vs_score.png",
        dpi=150,
    )

    plt.close()

    plot_small_tree(
        X_train,
        y_train,
        str(CHARTS_DIR / "day20_decision_tree.png"),
    )

    classification_rf = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
    )

    classification_rf.fit(
        X_train,
        y_train,
    )

    plot_feature_importance(
        classification_rf,
        list(X_train.columns),
        str(
            CHARTS_DIR
            / "day20_classification_feature_importance.png"
        ),
    )

    (
        regression_X_train,
        regression_X_test,
        regression_y_train,
        regression_y_test,
    ) = load_regression_data()

    regression_rf = RandomForestRegressor(
        n_estimators=100,
        random_state=42,
    )

    regression_rf.fit(
        regression_X_train,
        regression_y_train,
    )

    plot_feature_importance(
        regression_rf,
        list(regression_X_train.columns),
        str(
            CHARTS_DIR
            / "day20_regression_feature_importance.png"
        ),
    )

    classification_comparison = train_and_compare_models(
        "classification"
    )

    regression_comparison = train_and_compare_models(
        "regression"
    )

    print("\nClassification Model Comparison:")
    print(classification_comparison)

    print("\nRegression Model Comparison:")
    print(regression_comparison)


if __name__ == "__main__":
    main()