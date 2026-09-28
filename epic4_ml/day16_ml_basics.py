import pandas as pd
from sklearn.datasets import fetch_california_housing, load_breast_cancer
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# Load and save the required datasets
california = fetch_california_housing(as_frame=True)
breast_cancer = load_breast_cancer(as_frame=True)

california.frame.to_csv(
    "epic4_ml/data/california_housing.csv",
    index=False
)

breast_cancer.frame.to_csv(
    "epic4_ml/data/breast_cancer.csv",
    index=False
)


def load_ml_dataset(name: str) -> tuple[pd.DataFrame, pd.Series]:
    if name == "california_housing":
        data = pd.read_csv("epic4_ml/data/california_housing.csv")
        X = data.drop(columns=["MedHouseVal"])
        y = data["MedHouseVal"]
        return X, y

    if name == "breast_cancer":
        data = pd.read_csv("epic4_ml/data/breast_cancer.csv")
        X = data.drop(columns=["target"])
        y = data["target"]
        return X, y

    raise ValueError("Unknown dataset name")


def split_data(
    X,
    y,
    test_size: float,
    stratify: bool
) -> tuple:
    if len(X) == 0 or len(y) == 0:
        raise ValueError("Input data cannot be empty")

    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1")

    stratify_data = y if stratify else None

    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=42,
        stratify=stratify_data,
    )


def get_baseline_scores(
    X_train,
    X_test,
    y_train,
    y_test,
    task: str
) -> dict:
    if task == "regression":
        model = DummyRegressor(strategy="mean")
        model.fit(X_train, y_train)
        score = model.score(X_test, y_test)
        return {"baseline_score": score}

    if task == "classification":
        model = DummyClassifier(strategy="most_frequent")
        model.fit(X_train, y_train)
        score = model.score(X_test, y_test)
        return {"baseline_score": score}

    raise ValueError("task must be regression or classification")


def demonstrate_leakage(X, y) -> dict:
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
    )

    # Leakage: scaler is fitted using the complete dataset before the split.
    scaler_full = StandardScaler()
    X_scaled_full = scaler_full.fit_transform(X)

    X_train_full = X_scaled_full[X_train.index]
    X_test_full = X_scaled_full[X_test.index]

    # Correct approach: scaler is fitted only on training data.
    # The test data is transformed using the scaler learned from training data.
    scaler_train = StandardScaler()
    X_train_scaled = scaler_train.fit_transform(X_train)
    X_test_scaled = scaler_train.transform(X_test)

    return {
        "leakage_train_mean": float(X_train_full.mean()),
        "leakage_test_mean": float(X_test_full.mean()),
        "no_leakage_train_mean": float(X_train_scaled.mean()),
        "no_leakage_test_mean": float(X_test_scaled.mean()),
    }


class MLExperiment:
    def __init__(
        self,
        dataset_name: str,
        train_size: int,
        test_size: int,
        baseline_score: float,
    ):
        self.dataset_name = dataset_name
        self.train_size = train_size
        self.test_size = test_size
        self.baseline_score = baseline_score

    def summary(self) -> dict:
        return {
            "dataset_name": self.dataset_name,
            "train_size": self.train_size,
            "test_size": self.test_size,
            "baseline_score": self.baseline_score,
        }


# Overfitting means a model learns the training data too closely and performs
# poorly on new data. Underfitting means the model is too simple to learn
# important patterns. Bias is error from a model being too simple, while
# variance is error from a model changing too much with different training data.


california_X, california_y = load_ml_dataset("california_housing")
breast_cancer_X, breast_cancer_y = load_ml_dataset("breast_cancer")

california_X_train, california_X_test, california_y_train, california_y_test = (
    split_data(
        california_X,
        california_y,
        test_size=0.2,
        stratify=False,
    )
)

(
    breast_cancer_X_train,
    breast_cancer_X_test,
    breast_cancer_y_train,
    breast_cancer_y_test,
) = split_data(
    breast_cancer_X,
    breast_cancer_y,
    test_size=0.2,
    stratify=True,
)


print("California Housing:", california_X.shape, california_y.shape)
print("Breast Cancer:", breast_cancer_X.shape, breast_cancer_y.shape)

print(
    "California split:",
    california_X_train.shape,
    california_X_test.shape,
)

print(
    "Breast Cancer split:",
    breast_cancer_X_train.shape,
    breast_cancer_X_test.shape,
)


print("\nBreast Cancer class distribution before split:")
print(breast_cancer_y.value_counts(normalize=True))

print("\nBreast Cancer class distribution in training set:")
print(breast_cancer_y_train.value_counts(normalize=True))

print("\nBreast Cancer class distribution in test set:")
print(breast_cancer_y_test.value_counts(normalize=True))


california_baseline = get_baseline_scores(
    california_X_train,
    california_X_test,
    california_y_train,
    california_y_test,
    "regression",
)

breast_cancer_baseline = get_baseline_scores(
    breast_cancer_X_train,
    breast_cancer_X_test,
    breast_cancer_y_train,
    breast_cancer_y_test,
    "classification",
)


print("\nCalifornia Housing baseline:")
print(california_baseline)

print("\nBreast Cancer baseline:")
print(breast_cancer_baseline)


print("\nCalifornia leakage demonstration:")
print(demonstrate_leakage(california_X, california_y))


california_experiment = MLExperiment(
    "california_housing",
    len(california_X_train),
    len(california_X_test),
    california_baseline["baseline_score"],
)

breast_cancer_experiment = MLExperiment(
    "breast_cancer",
    len(breast_cancer_X_train),
    len(breast_cancer_X_test),
    breast_cancer_baseline["baseline_score"],
)


print("\nCalifornia Experiment:")
print(california_experiment.summary())

print("\nBreast Cancer Experiment:")
print(breast_cancer_experiment.summary())