import numpy as np

from epic5_advanced_ml.day22_clustering_pca import (
    find_optimal_k,
    run_kmeans,
    apply_pca,
)


def test_find_optimal_k_returns_requested_k_values():
    X = np.array([
        [1.0, 1.0],
        [1.2, 1.1],
        [5.0, 5.0],
        [5.2, 5.1],
        [9.0, 9.0],
        [9.2, 9.1],
    ])

    results = find_optimal_k(X, range(2, 5))

    assert set(results.keys()) == {2, 3, 4}


def test_inertia_decreases_as_k_increases():
    X = np.array([
        [1.0, 1.0],
        [1.2, 1.1],
        [5.0, 5.0],
        [5.2, 5.1],
        [9.0, 9.0],
        [9.2, 9.1],
    ])

    results = find_optimal_k(X, range(2, 5))
    inertias = [results[k]["inertia"] for k in range(2, 5)]

    assert inertias[0] >= inertias[1] >= inertias[2]


def test_silhouette_score_is_in_valid_range():
    X = np.array([
        [1.0, 1.0],
        [1.2, 1.1],
        [5.0, 5.0],
        [5.2, 5.1],
        [9.0, 9.0],
        [9.2, 9.1],
    ])

    results = find_optimal_k(X, range(2, 5))

    for result in results.values():
        assert -1 <= result["silhouette"] <= 1


def test_run_kmeans_returns_correct_labels():
    X = np.array([
        [1.0, 1.0],
        [1.2, 1.1],
        [5.0, 5.0],
        [5.2, 5.1],
    ])

    labels, model = run_kmeans(X, 2)

    assert len(labels) == len(X)
    assert model.n_clusters == 2


def test_run_kmeans_returns_expected_number_of_clusters():
    X = np.array([
        [1.0, 1.0],
        [1.2, 1.1],
        [5.0, 5.0],
        [5.2, 5.1],
    ])

    labels, _ = run_kmeans(X, 2)

    assert len(np.unique(labels)) == 2


def test_pca_output_shape():
    X = np.random.RandomState(42).rand(20, 4)

    X_pca, pca = apply_pca(X, 2)

    assert X_pca.shape == (20, 2)
    assert pca.n_components == 2


def test_pca_explained_variance_is_at_most_one():
    X = np.random.RandomState(42).rand(20, 4)

    _, pca = apply_pca(X, 2)

    explained_variance = pca.explained_variance_ratio_.sum()

    assert explained_variance <= 1.0


def test_pca_explained_variance_is_non_negative():
    X = np.random.RandomState(42).rand(20, 4)

    _, pca = apply_pca(X, 2)

    assert np.all(pca.explained_variance_ratio_ >= 0)