import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path

from sklearn.ensemble import (
    BaggingClassifier,
    BaggingRegressor,
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
    StackingClassifier,
    StackingRegressor,
)
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import accuracy_score, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from xgboost import XGBClassifier, XGBRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
from lightgbm import early_stopping


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "epic4_ml" / "data"

CLASSIFICATION_DATA = DATA_DIR / "breast_cancer.csv"
REGRESSION_DATA = DATA_DIR / "california_housing.csv"


def load_classification_data():
    df = pd.read_csv(CLASSIFICATION_DATA)

    X = df.drop(columns=["target"])
    y = df["target"]

    return X, y


def load_regression_data():
    df = pd.read_csv(REGRESSION_DATA)

    X = df.drop(columns=["MedHouseVal"])
    y = df["MedHouseVal"]

    return X, y


def split_classification_data(X, y):
    return train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )


def split_regression_data(X, y):
    return train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
    )


def split_validation_data(X_train, y_train, task):
    if task == "classification":
        return train_test_split(
            X_train,
            y_train,
            test_size=0.2,
            random_state=42,
            stratify=y_train,
        )

    return train_test_split(
        X_train,
        y_train,
        test_size=0.2,
        random_state=42,
    )


def train_bagging_models(X_train, y_train, task):
    if task == "classification":
        bagging_model = BaggingClassifier(
            n_estimators=100,
            random_state=42,
        )

        random_forest_model = RandomForestClassifier(
            n_estimators=100,
            random_state=42,
        )

    else:
        bagging_model = BaggingRegressor(
            n_estimators=100,
            random_state=42,
        )

        random_forest_model = RandomForestRegressor(
            n_estimators=100,
            random_state=42,
        )

    bagging_model.fit(X_train, y_train)
    random_forest_model.fit(X_train, y_train)

    return {
        "Bagging": bagging_model,
        "Random Forest": random_forest_model,
    }


def train_gradient_boosting_model(X_train, y_train, task):
    if task == "classification":
        model = GradientBoostingClassifier(
            n_estimators=100,
            random_state=42,
        )
    else:
        model = GradientBoostingRegressor(
            n_estimators=100,
            random_state=42,
        )

    model.fit(X_train, y_train)

    return model


def train_boosting_models(
    X_train,
    y_train,
    X_val,
    y_val,
    task,
):
    models = {}
    histories = {}

    if task == "classification":
        xgb_model = XGBClassifier(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=4,
            random_state=42,
            eval_metric="logloss",
            early_stopping_rounds=20,
        )

        lgbm_model = LGBMClassifier(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=4,
            random_state=42,
            verbosity=-1,
        )

        xgb_model.fit(
            X_train,
            y_train,
            eval_set=[
                (X_train, y_train),
                (X_val, y_val),
            ],
            verbose=False,
        )

        lgbm_model.fit(
            X_train,
            y_train,
            eval_set=[
                (X_train, y_train),
                (X_val, y_val),
            ],
            callbacks=[
                early_stopping(
                    stopping_rounds=10,
                    min_delta=0.001,
                    verbose=False,
                )
            ],
        )

    else:
        xgb_model = XGBRegressor(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=4,
            random_state=42,
            eval_metric="rmse",
            early_stopping_rounds=20,
        )

        lgbm_model = LGBMRegressor(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=4,
            random_state=42,
            verbosity=-1,
        )

        xgb_model.fit(
            X_train,
            y_train,
            eval_set=[
                (X_train, y_train),
                (X_val, y_val),
            ],
            verbose=False,
        )

        lgbm_model.fit(
            X_train,
            y_train,
            eval_set=[
                (X_train, y_train),
                (X_val, y_val),
            ],
            callbacks=[
                early_stopping(
                    stopping_rounds=10,
                    min_delta=0.001,
                    verbose=False,
                )
            ],
        )

    models["XGBoost"] = xgb_model
    models["LightGBM"] = lgbm_model

    histories["XGBoost"] = xgb_model.evals_result()
    histories["LightGBM"] = lgbm_model.evals_result_

    return models, histories


def plot_training_curves(histories, save_path):
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(12, 5),
    )

    for ax, model_name in zip(
        axes,
        ["XGBoost", "LightGBM"],
    ):
        history = histories[model_name]

        if model_name == "XGBoost":
            train_key = "validation_0"
            valid_key = "validation_1"
        else:
            train_key = "training"
            valid_key = "valid_1"

        metric = list(
            history[train_key].keys()
        )[0]

        ax.plot(
            history[train_key][metric],
            label="Training",
        )

        ax.plot(
            history[valid_key][metric],
            label="Validation",
        )

        ax.set_title(
            f"{model_name} Training vs Validation"
        )
        ax.set_xlabel("Boosting Round")
        ax.set_ylabel("Loss")
        ax.legend()

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def build_stacking_model(
    base_models,
    meta_model,
    task,
):
    if task == "classification":
        return StackingClassifier(
            estimators=base_models,
            final_estimator=meta_model,
        )

    return StackingRegressor(
        estimators=base_models,
        final_estimator=meta_model,
    )


def compare_feature_importance(
    models,
    feature_names,
):
    importance_data = {}

    for model_name, model in models.items():
        if hasattr(model, "feature_importances_"):
            importance_data[model_name] = (
                model.feature_importances_
            )

    # Random Forest, XGBoost, and LightGBM
    # can produce different feature rankings.
    return pd.DataFrame(
        importance_data,
        index=feature_names,
    )


def evaluate_models(
    models,
    X_test,
    y_test,
    task,
):
    results = []

    for model_name, model in models.items():
        predictions = model.predict(X_test)

        if task == "classification":
            score = accuracy_score(
                y_test,
                predictions,
            )
            metric = "Accuracy"
        else:
            score = r2_score(
                y_test,
                predictions,
            )
            metric = "R2"

        results.append(
            {
                "Model": model_name,
                "Task": task,
                "Metric": metric,
                "Score": score,
            }
        )

    return pd.DataFrame(results)


def run_classification():
    X, y = load_classification_data()

    X_train, X_test, y_train, y_test = (
        split_classification_data(X, y)
    )

    X_train_main, X_val, y_train_main, y_val = (
        split_validation_data(
            X_train,
            y_train,
            "classification",
        )
    )

    ensemble_models = train_bagging_models(
        X_train,
        y_train,
        "classification",
    )

    gradient_model = train_gradient_boosting_model(
        X_train,
        y_train,
        "classification",
    )

    boosting_models, histories = (
        train_boosting_models(
            X_train_main,
            y_train_main,
            X_val,
            y_val,
            "classification",
        )
    )

    ensemble_models["Gradient Boosting"] = (
        gradient_model
    )

    ensemble_models.update(boosting_models)

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

    stacking_model = build_stacking_model(
        base_models,
        LogisticRegression(
            max_iter=1000,
            random_state=42,
        ),
        "classification",
    )

    stacking_model.fit(
        X_train,
        y_train,
    )

    ensemble_models["Stacking"] = stacking_model

    results = evaluate_models(
        ensemble_models,
        X_test,
        y_test,
        "classification",
    )

    return (
        ensemble_models,
        histories,
        results,
    )


def run_regression():
    X, y = load_regression_data()

    X_train, X_test, y_train, y_test = (
        split_regression_data(X, y)
    )

    X_train_main, X_val, y_train_main, y_val = (
        split_validation_data(
            X_train,
            y_train,
            "regression",
        )
    )

    ensemble_models = train_bagging_models(
        X_train,
        y_train,
        "regression",
    )

    gradient_model = train_gradient_boosting_model(
        X_train,
        y_train,
        "regression",
    )

    boosting_models, histories = (
        train_boosting_models(
            X_train_main,
            y_train_main,
            X_val,
            y_val,
            "regression",
        )
    )

    ensemble_models["Gradient Boosting"] = (
        gradient_model
    )

    ensemble_models.update(boosting_models)

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

    stacking_model = build_stacking_model(
        base_models,
        LinearRegression(),
        "regression",
    )

    stacking_model.fit(
        X_train,
        y_train,
    )

    ensemble_models["Stacking"] = stacking_model

    results = evaluate_models(
        ensemble_models,
        X_test,
        y_test,
        "regression",
    )

    return (
        ensemble_models,
        histories,
        results,
    )


if __name__ == "__main__":
    (
        classification_models,
        classification_histories,
        classification_results,
    ) = run_classification()

    (
        regression_models,
        regression_histories,
        regression_results,
    ) = run_regression()

    classification_curve_path = (
        Path(__file__).parent
        / "day23_xgboost_lightgbm_classification.png"
    )

    regression_curve_path = (
        Path(__file__).parent
        / "day23_xgboost_lightgbm_regression.png"
    )

    plot_training_curves(
        classification_histories,
        classification_curve_path,
    )

    plot_training_curves(
        regression_histories,
        regression_curve_path,
    )

    classification_results = (
        classification_results.sort_values(
            by="Score",
            ascending=False,
        )
    )

    regression_results = (
        regression_results.sort_values(
            by="Score",
            ascending=False,
        )
    )

    print(
        "\nClassification Model Comparison:"
    )
    print(
        classification_results.to_string(
            index=False
        )
    )

    print(
        "\nRegression Model Comparison:"
    )
    print(
        regression_results.to_string(
            index=False
        )
    )

    classification_feature_importance = (
        compare_feature_importance(
            classification_models,
            load_classification_data()[
                0
            ].columns.tolist(),
        )
    )

    regression_feature_importance = (
        compare_feature_importance(
            regression_models,
            load_regression_data()[
                0
            ].columns.tolist(),
        )
    )

    print(
        "\nClassification Feature Importance:"
    )
    print(
        classification_feature_importance
    )

    print(
        "\nRegression Feature Importance:"
    )
    print(
        regression_feature_importance
    )