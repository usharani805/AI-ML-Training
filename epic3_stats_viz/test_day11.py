import numpy as np
import pandas as pd
from scipy import stats

from day11_descriptive_stats import (
    compute_central_tendency,
    compute_spread,
    conditional_probability,
    bayes_theorem,
)


def test_mean():
    data = np.array([10, 20, 30, 40, 50])
    result = compute_central_tendency(data)
    assert np.isclose(result["mean"], np.mean(data), atol=1e-6)


def test_median():
    data = np.array([10, 20, 30, 40, 50])
    result = compute_central_tendency(data)
    assert np.isclose(result["median"], np.median(data), atol=1e-6)


def test_mode():
    data = np.array([10, 20, 20, 30, 40])
    result = compute_central_tendency(data)
    assert np.isclose(result["mode"], stats.mode(data, keepdims=False).mode, atol=1e-6)


def test_variance():
    data = np.array([10, 20, 30, 40, 50])
    result = compute_spread(data)
    assert np.isclose(result["variance"], np.var(data, ddof=0), atol=1e-6)


def test_standard_deviation():
    data = np.array([10, 20, 30, 40, 50])
    result = compute_spread(data)
    assert np.isclose(result["std"], np.std(data, ddof=0), atol=1e-6)


def test_range():
    data = np.array([10, 20, 30, 40, 50])
    result = compute_spread(data)
    assert np.isclose(result["range"], np.ptp(data), atol=1e-6)


def test_iqr():
    data = np.array([10, 20, 30, 40, 50])
    result = compute_spread(data)
    assert np.isclose(result["iqr"], stats.iqr(data), atol=1e-6)


def test_conditional_probability():
    df = pd.DataFrame(
        {
            "High_Salary": [True, True, False, True, False],
            "IT_Department": [True, False, True, True, False],
        }
    )

    result = conditional_probability(
        df,
        "High_Salary",
        "IT_Department",
    )

    assert np.isclose(result, 2 / 3, atol=1e-6)


def test_bayes_theorem():
    result = bayes_theorem(
        p_a=0.2,
        p_b_given_a=0.8,
        p_b=0.3,
    )

    expected = (0.8 * 0.2) / 0.3

    assert np.isclose(result, expected, atol=1e-6)