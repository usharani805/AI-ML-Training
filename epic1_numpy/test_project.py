import numpy as np

from epic1_numpy.project_student_analytics import (
    generate_scores,
    compute_subject_stats,
    rank_students,
    identify_at_risk_students,
    normalize_scores,
    correlation_between_subjects,
    save_report,
    main
)


def test_generate_scores_shape():
    scores = generate_scores()

    assert scores.shape == (200, 5)


def test_generate_scores_range():
    scores = generate_scores()

    assert np.all(scores >= 0)
    assert np.all(scores <= 100)


def test_mean():
    scores = np.array([
        [10, 20, 30],
        [20, 30, 40]
    ])

    stats = compute_subject_stats(scores)

    assert np.array_equal(stats["mean"], [15, 25, 35])


def test_median():
    scores = np.array([
        [10, 20, 30],
        [20, 30, 40]
    ])

    stats = compute_subject_stats(scores)

    assert np.array_equal(stats["median"], [15, 25, 35])


def test_std():
    scores = np.array([
        [10, 20],
        [20, 30]
    ])

    stats = compute_subject_stats(scores)

    assert np.allclose(
        stats["std"],
        [5, 5]
    )


def test_min():
    scores = np.array([
        [10, 20, 30],
        [20, 30, 40]
    ])

    stats = compute_subject_stats(scores)

    assert np.array_equal(stats["min"], [10, 20, 30])


def test_max():
    scores = np.array([
        [10, 20, 30],
        [20, 30, 40]
    ])

    stats = compute_subject_stats(scores)

    assert np.array_equal(stats["max"], [20, 30, 40])


def test_ranking():
    scores = np.array([
        [50, 50],
        [90, 90],
        [70, 70]
    ])

    ranking = rank_students(scores)

    assert np.array_equal(ranking, [1, 2, 0])


def test_at_risk_students():
    scores = np.array([
        [30, 30],
        [80, 80],
        [40, 40]
    ])

    result = identify_at_risk_students(scores, 50)

    assert np.array_equal(result, [0, 2])


def test_normalization_shape():
    scores = np.array([
        [10, 20],
        [20, 30],
        [30, 40]
    ])

    result = normalize_scores(scores)

    assert result.shape == scores.shape


def test_normalization_mean():
    scores = np.array([
        [10, 20],
        [20, 30],
        [30, 40]
    ])

    result = normalize_scores(scores)

    assert np.allclose(
        np.mean(result, axis=0),
        [0, 0]
    )


def test_correlation_shape():
    scores = np.array([
        [10, 20, 30],
        [20, 30, 40],
        [30, 40, 50]
    ])

    result = correlation_between_subjects(scores)

    assert result.shape == (3, 3)


def test_correlation_symmetry():
    scores = np.array([
        [10, 20, 30],
        [20, 30, 40],
        [30, 40, 50]
    ])

    result = correlation_between_subjects(scores)

    assert np.allclose(result, result.T)


def test_save_report(tmp_path):
    report_file = tmp_path / "test_report.txt"

    stats = {
        "mean": np.array([10, 20, 30])
    }

    save_report(stats, str(report_file))

    assert report_file.exists()


def test_main(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    main()

    assert (tmp_path / "student_analytics_report.txt").exists()