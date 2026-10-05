import os
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


def euclidean_distance_matrix(
    X1: np.ndarray,
    X2: np.ndarray,
) -> np.ndarray:
    """Calculate Euclidean distances between two sets of samples."""
    return np.sqrt(
        np.sum(
            (X1[:, np.newaxis, :] - X2[np.newaxis, :, :]) ** 2,
            axis=2,
        )
    )


def knn_predict_from_scratch(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    k: int,
) -> np.ndarray:
    """Predict classes using KNN implemented from scratch."""
    if k <= 0:
        raise ValueError("k must be greater than 0")

    if k > len(X_train):
        raise ValueError(
            "k cannot be greater than the number of training samples"
        )

    distances = euclidean_distance_matrix(X_test, X_train)
    predictions = []

    for distance_row in distances:
        nearest_indices = np.argsort(distance_row)[:k]
        nearest_labels = y_train[nearest_indices]

        labels, counts = np.unique(
            nearest_labels,
            return_counts=True,
        )

        predictions.append(labels[np.argmax(counts)])

    return np.array(predictions)


def k_sweep_analysis(
    X_train,
    X_test,
    y_train,
    y_test,
    k_range: range,
) -> pd.DataFrame:
    """Evaluate sklearn KNN for different k values."""
    results = []

    for k in k_range:
        model = KNeighborsClassifier(n_neighbors=k)
        model.fit(X_train, y_train)

        predictions = model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)

        results.append(
            {
                "k": k,
                "accuracy": accuracy,
            }
        )

    return pd.DataFrame(results)


def plot_svm_decision_boundary(
    model,
    X: np.ndarray,
    y: np.ndarray,
    save_path: str,
) -> None:
    """Plot and save an SVM decision boundary."""
    x_min = X[:, 0].min() - 1
    x_max = X[:, 0].max() + 1
    y_min = X[:, 1].min() - 1
    y_max = X[:, 1].max() + 1

    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 300),
        np.linspace(y_min, y_max, 300),
    )

    grid = np.c_[xx.ravel(), yy.ravel()]
    predictions = model.predict(grid).reshape(xx.shape)

    plt.figure(figsize=(8, 6))

    plt.contourf(
        xx,
        yy,
        predictions,
        alpha=0.3,
    )

    plt.scatter(
        X[:, 0],
        X[:, 1],
        c=y,
        edgecolors="k",
    )

    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.title(
        f"SVM Decision Boundary - {model.kernel} Kernel"
    )

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def compare_kernels(
    X_train,
    X_test,
    y_train,
    y_test,
    kernels: list,
) -> pd.DataFrame:
    """Compare SVM kernels using accuracy, F1 and training time."""
    results = []

    for kernel in kernels:
        start_time = time.perf_counter()

        model = SVC(kernel=kernel)
        model.fit(X_train, y_train)

        training_time = time.perf_counter() - start_time

        predictions = model.predict(X_test)

        results.append(
            {
                "kernel": kernel,
                "accuracy": accuracy_score(
                    y_test,
                    predictions,
                ),
                "f1": f1_score(
                    y_test,
                    predictions,
                ),
                "training_time": training_time,
            }
        )

    return pd.DataFrame(results)


def main():
    data_path = "epic4_ml/data/breast_cancer.csv"
    output_dir = "epic5_advanced_ml/charts"

    os.makedirs(output_dir, exist_ok=True)

    data = pd.read_csv(data_path)

    X = data.drop(columns=["target"]).values
    y = data["target"].values

    from sklearn.model_selection import train_test_split

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

    # KNN from scratch
    scratch_predictions = knn_predict_from_scratch(
        X_train_scaled,
        y_train,
        X_test_scaled,
        k=5,
    )

    sklearn_knn = KNeighborsClassifier(
        n_neighbors=5
    )

    sklearn_knn.fit(
        X_train_scaled,
        y_train,
    )

    sklearn_predictions = sklearn_knn.predict(
        X_test_scaled
    )

    match_percentage = (
        np.mean(
            scratch_predictions == sklearn_predictions
        )
        * 100
    )

    print("\nKNN Scratch vs sklearn")
    print(
        f"Prediction match: {match_percentage:.2f}%"
    )

    # K sweep
    k_results = k_sweep_analysis(
        X_train_scaled,
        X_test_scaled,
        y_train,
        y_test,
        range(1, 21),
    )

    print("\nK Sweep Results")
    print(
        k_results.to_string(index=False)
    )

    best_k = int(
        k_results.loc[
            k_results["accuracy"].idxmax(),
            "k",
        ]
    )

    print(f"\nBest k: {best_k}")

    plt.figure(figsize=(8, 6))

    plt.plot(
        k_results["k"],
        k_results["accuracy"],
        marker="o",
    )

    plt.xlabel("k")
    plt.ylabel("Accuracy")
    plt.title("KNN Accuracy vs k")
    plt.xticks(range(1, 21))
    plt.grid(True)
    plt.tight_layout()

    plt.savefig(
        os.path.join(
            output_dir,
            "knn_k_sweep_accuracy.png",
        )
    )

    plt.close()

    # Scaling comparison
    knn_without_scaling = KNeighborsClassifier(
        n_neighbors=best_k
    )

    knn_without_scaling.fit(
        X_train,
        y_train,
    )

    accuracy_without_scaling = (
        knn_without_scaling.score(
            X_test,
            y_test,
        )
    )

    knn_with_scaling = KNeighborsClassifier(
        n_neighbors=best_k
    )

    knn_with_scaling.fit(
        X_train_scaled,
        y_train,
    )

    accuracy_with_scaling = (
        knn_with_scaling.score(
            X_test_scaled,
            y_test,
        )
    )

    print("\nKNN Scaling Comparison")
    print(
        f"Without scaling: "
        f"{accuracy_without_scaling:.4f}"
    )
    print(
        f"With StandardScaler: "
        f"{accuracy_with_scaling:.4f}"
    )

    # SVM kernel comparison
    kernels = [
        "linear",
        "rbf",
        "poly",
    ]

    kernel_results = compare_kernels(
        X_train_scaled,
        X_test_scaled,
        y_train,
        y_test,
        kernels,
    )

    print("\nSVM Kernel Comparison")
    print(
        kernel_results.to_string(index=False)
    )

    # SVM decision boundaries
    X_2d = X[:, :2]

    X_2d_train, X_2d_test, y_2d_train, y_2d_test = (
        train_test_split(
            X_2d,
            y,
            test_size=0.2,
            random_state=42,
            stratify=y,
        )
    )

    scaler_2d = StandardScaler()

    X_2d_train_scaled = scaler_2d.fit_transform(
        X_2d_train
    )

    X_2d_test_scaled = scaler_2d.transform(
        X_2d_test
    )

    models = {}

    for kernel in kernels:
        model = SVC(kernel=kernel)

        model.fit(
            X_2d_train_scaled,
            y_2d_train,
        )

        models[kernel] = model

    x_min = X_2d_train_scaled[:, 0].min() - 1
    x_max = X_2d_train_scaled[:, 0].max() + 1
    y_min = X_2d_train_scaled[:, 1].min() - 1
    y_max = X_2d_train_scaled[:, 1].max() + 1

    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 300),
        np.linspace(y_min, y_max, 300),
    )

    grid = np.c_[
        xx.ravel(),
        yy.ravel(),
    ]

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(18, 5),
    )

    for ax, kernel in zip(
        axes,
        kernels,
    ):
        predictions = (
            models[kernel]
            .predict(grid)
            .reshape(xx.shape)
        )

        ax.contourf(
            xx,
            yy,
            predictions,
            alpha=0.3,
        )

        ax.scatter(
            X_2d_train_scaled[:, 0],
            X_2d_train_scaled[:, 1],
            c=y_2d_train,
            edgecolors="k",
        )

        ax.set_title(
            f"{kernel.upper()} Kernel"
        )

        ax.set_xlabel("Feature 1")
        ax.set_ylabel("Feature 2")

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            output_dir,
            "svm_kernel_decision_boundaries.png",
        )
    )

    plt.close()

    # Manual RBF tuning
    print("\nManual RBF Tuning")

    C_values = [
        0.1,
        1,
        10,
    ]

    gamma_values = [
        "scale",
        0.01,
        0.1,
        1,
    ]

    tuning_results = []

    for C in C_values:
        for gamma in gamma_values:
            model = SVC(
                kernel="rbf",
                C=C,
                gamma=gamma,
            )

            model.fit(
                X_train_scaled,
                y_train,
            )

            predictions = model.predict(
                X_test_scaled
            )

            tuning_results.append(
                {
                    "C": C,
                    "gamma": gamma,
                    "accuracy": accuracy_score(
                        y_test,
                        predictions,
                    ),
                    "f1": f1_score(
                        y_test,
                        predictions,
                    ),
                }
            )

    tuning_df = pd.DataFrame(
        tuning_results
    )

    print(
        tuning_df.to_string(index=False)
    )

    best_rbf = tuning_df.loc[
        tuning_df["accuracy"].idxmax()
    ]

    print("\nBest RBF Parameters")
    print(
        f"C: {best_rbf['C']}"
    )
    print(
        f"gamma: {best_rbf['gamma']}"
    )
    print(
        f"accuracy: "
        f"{best_rbf['accuracy']:.4f}"
    )


if __name__ == "__main__":
    main()