import numpy as np
import pandas as pd
import pytest
from scipy import stats

from day12_correlation_distributions import (
    pearson_correlation,
    correlation_matrix_report,
    fit_normal_distribution,
    compute_zscores,
)


def test_pearson_matches_scipy():
    x = np.array([1, 2, 3, 4, 5])
    y = np.array([2, 4, 6, 8, 10])

    result = pearson_correlation(x, y)
    expected = stats.pearsonr(x, y).statistic

    assert np.isclose(result, expected)


def test_pearson_negative_correlation():
    x = np.array([1, 2, 3, 4, 5])
    y = np.array([10, 8, 6, 4, 2])

    result = pearson_correlation(x, y)

    assert np.isclose(result, -1.0)


def test_pearson_mismatched_lengths():
    x = np.array([1, 2, 3])
    y = np.array([1, 2])

    with pytest.raises(ValueError):
        pearson_correlation(x, y)


def test_correlation_matrix_report():
    df = pd.DataFrame(
        {
            "Age": [20, 30, 40, 50],
            "Salary": [20000, 30000, 40000, 50000],
            "Orders": [1, 2, 3, 4],
        }
    )

    result = correlation_matrix_report(df)

    assert len(result) == 3
    assert "Pearson" in result.columns
    assert "Spearman" in result.columns


def test_fit_normal_distribution():
    data = np.array([10, 20, 30, 40, 50])

    result = fit_normal_distribution(data)

    assert "mean" in result
    assert "std" in result
    assert "shapiro_p_value" in result
    assert "normally_distributed" in result


def test_zscores_mean_is_zero():
    data = np.array([1, 2, 3, 4, 5])

    result = compute_zscores(data)

    assert np.isclose(np.mean(result), 0.0)


def test_zscore_outlier_flagging():
    data = np.array([10, 11, 10, 9, 10, 100])

    zscores = compute_zscores(data)
    outliers = data[np.abs(zscores) > 2]

    assert 100 in outliers