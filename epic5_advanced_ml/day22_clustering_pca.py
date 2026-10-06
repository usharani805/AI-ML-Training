import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from scipy.cluster.hierarchy import dendrogram, linkage


def find_optimal_k(X: np.ndarray, k_range: range) -> dict:
    """Return inertia and silhouette score for each k."""
    results = {}

    for k in k_range:
        model = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = model.fit_predict(X)

        results[k] = {
            "inertia": model.inertia_,
            "silhouette": silhouette_score(X, labels),
        }

    return results


def run_kmeans(
    X: np.ndarray, n_clusters: int
) -> tuple[np.ndarray, KMeans]:
    """Run KMeans clustering and return labels and fitted model."""
    model = KMeans(
        n_clusters=n_clusters,
        random_state=42,
        n_init=10,
    )
    labels = model.fit_predict(X)

    return labels, model


def plot_dendrogram(X: np.ndarray, save_path: str) -> None:
    """Create and save a hierarchical clustering dendrogram."""
    linkage_matrix = linkage(X, method="ward")

    plt.figure(figsize=(10, 6))
    dendrogram(linkage_matrix)
    plt.title("Hierarchical Clustering Dendrogram")
    plt.xlabel("Samples")
    plt.ylabel("Distance")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def apply_pca(
    X: np.ndarray, n_components: int
) -> tuple[np.ndarray, PCA]:
    """Apply PCA and return transformed data and fitted PCA model."""
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X)

    return X_pca, pca


if __name__ == "__main__":
    # Load Iris dataset with known true labels.
    iris = load_iris()
    X = iris.data
    y_true = iris.target

    # Scale features before clustering.
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # PCA before clustering:
    # Use PCA before clustering when there are many features,
    # strong correlation, noise, or when visualization is needed.
    #
    # Do not use PCA before clustering when the dataset already has
    # few useful features and dimensionality reduction may remove
    # important information.

    # Find optimal k using elbow and silhouette scores.
    k_results = find_optimal_k(X_scaled, range(2, 8))

    print("\nKMeans Results:")
    print(pd.DataFrame.from_dict(k_results, orient="index"))

    # Elbow plot.
    ks = list(k_results.keys())
    inertias = [k_results[k]["inertia"] for k in ks]

    plt.figure(figsize=(8, 5))
    plt.plot(ks, inertias, marker="o")
    plt.xlabel("Number of Clusters (k)")
    plt.ylabel("Inertia")
    plt.title("KMeans Elbow Method")
    plt.tight_layout()
    plt.savefig("epic5_advanced_ml/day22_elbow.png")
    plt.close()

    # Silhouette score plot.
    silhouettes = [k_results[k]["silhouette"] for k in ks]

    plt.figure(figsize=(8, 5))
    plt.plot(ks, silhouettes, marker="o")
    plt.xlabel("Number of Clusters (k)")
    plt.ylabel("Silhouette Score")
    plt.title("Silhouette Scores")
    plt.tight_layout()
    plt.savefig("epic5_advanced_ml/day22_silhouette.png")
    plt.close()

    # Choose optimal k using the highest silhouette score.
    optimal_k = max(
        k_results,
        key=lambda k: k_results[k]["silhouette"],
    )

    print(f"\nOptimal k: {optimal_k}")

    # KMeans clustering.
    kmeans_labels, kmeans_model = run_kmeans(
        X_scaled,
        optimal_k,
    )

    # DBSCAN clustering.
    dbscan = DBSCAN(
        eps=0.7,
        min_samples=5,
    )
    dbscan_labels = dbscan.fit_predict(X_scaled)

    dbscan_clusters = len(set(dbscan_labels)) - (
        1 if -1 in dbscan_labels else 0
    )
    dbscan_noise = np.sum(dbscan_labels == -1)

    print(f"DBSCAN clusters: {dbscan_clusters}")
    print(f"DBSCAN noise points: {dbscan_noise}")

    # Agglomerative clustering.
    agglomerative = AgglomerativeClustering(
        n_clusters=optimal_k
    )
    agglomerative_labels = agglomerative.fit_predict(X_scaled)

    print("\nClustering comparison:")
    print(
        f"KMeans clusters: "
        f"{len(np.unique(kmeans_labels))}"
    )
    print(
        f"Agglomerative clusters: "
        f"{len(np.unique(agglomerative_labels))}"
    )
    print(
        "DBSCAN can identify noise points with label -1, "
        "while KMeans assigns every point to a cluster."
    )

    # Hierarchical clustering dendrogram.
    plot_dendrogram(
        X_scaled,
        "epic5_advanced_ml/day22_dendrogram.png",
    )

    # PCA.
    X_pca, pca = apply_pca(
        X_scaled,
        2,
    )

    print(
        "\nPCA explained variance ratio:",
        pca.explained_variance_ratio_,
    )
    print(
        "Total explained variance:",
        pca.explained_variance_ratio_.sum(),
    )

    # PCA scree plot.
    pca_full = PCA()
    pca_full.fit(X_scaled)

    plt.figure(figsize=(8, 5))
    plt.plot(
        range(
            1,
            len(pca_full.explained_variance_ratio_) + 1,
        ),
        pca_full.explained_variance_ratio_,
        marker="o",
    )
    plt.xlabel("Principal Component")
    plt.ylabel("Explained Variance Ratio")
    plt.title("PCA Scree Plot")
    plt.tight_layout()
    plt.savefig("epic5_advanced_ml/day22_pca_scree.png")
    plt.close()

    # Transform KMeans cluster centers into PCA space.
    centers_pca = pca.transform(
        kmeans_model.cluster_centers_
    )

    # 2D visualization:
    # True labels vs assigned KMeans clusters.
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.scatter(
        X_pca[:, 0],
        X_pca[:, 1],
        c=y_true,
        cmap="viridis",
    )
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("True Labels")

    plt.subplot(1, 2, 2)
    plt.scatter(
        X_pca[:, 0],
        X_pca[:, 1],
        c=kmeans_labels,
        cmap="viridis",
    )
    plt.scatter(
        centers_pca[:, 0],
        centers_pca[:, 1],
        marker="X",
        s=200,
        edgecolors="black",
    )
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("KMeans Clusters")

    plt.tight_layout()
    plt.savefig(
        "epic5_advanced_ml/day22_pca_clusters.png"
    )
    plt.close()

    print("\nDay 22 completed successfully.")