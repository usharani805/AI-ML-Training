import numpy as np
import pandas as pd
from scipy import stats


def compute_central_tendency(data: np.ndarray) -> dict:
    """Compute mean, median, and mode from scratch using NumPy."""
    data = np.asarray(data, dtype=float)

    mean = np.sum(data) / len(data)
    median = np.median(data)

    values, counts = np.unique(data, return_counts=True)
    mode = values[np.argmax(counts)]

    return {
        "mean": mean,
        "median": median,
        "mode": mode,
    }


def compute_spread(data: np.ndarray) -> dict:
    """Compute variance, standard deviation, range, and IQR."""
    data = np.asarray(data, dtype=float)

    mean = np.sum(data) / len(data)
    variance = np.sum((data - mean) ** 2) / len(data)
    standard_deviation = np.sqrt(variance)

    data_range = np.max(data) - np.min(data)

    q1 = np.percentile(data, 25)
    q3 = np.percentile(data, 75)
    iqr = q3 - q1

    return {
        "variance": variance,
        "std": standard_deviation,
        "range": data_range,
        "iqr": iqr,
    }


def conditional_probability(
    df: pd.DataFrame, event_a: str, event_b: str
) -> float:
    """Calculate P(A|B) from boolean event columns."""
    event_b_count = df[event_b].sum()

    if event_b_count == 0:
        raise ValueError("Event B has zero occurrences.")

    event_a_and_b_count = (df[event_a] & df[event_b]).sum()

    return event_a_and_b_count / event_b_count


def bayes_theorem(
    p_a: float, p_b_given_a: float, p_b: float
) -> float:
    """Calculate P(A|B) using Bayes' theorem."""
    if p_b == 0:
        raise ValueError("P(B) cannot be zero.")

    return (p_b_given_a * p_a) / p_b


def pearson_correlation(x: np.ndarray, y: np.ndarray) -> float:
    """Calculate Pearson correlation coefficient from scratch."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    if len(x) != len(y):
        raise ValueError("x and y must have the same length.")

    x_mean = np.mean(x)
    y_mean = np.mean(y)

    numerator = np.sum((x - x_mean) * (y - y_mean))
    denominator = np.sqrt(
        np.sum((x - x_mean) ** 2) * np.sum((y - y_mean) ** 2)
    )

    if denominator == 0:
        raise ValueError("Correlation is undefined for constant data.")

    return numerator / denominator


def correlation_matrix_report(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create a full pairwise correlation report for numeric columns.

    Pearson measures linear relationships.
    Spearman is preferred when the relationship is based on ranks,
    is monotonic but not necessarily linear, or when outliers may
    affect Pearson correlation.
    """
    numeric_df = df.select_dtypes(include=np.number)

    rows = []

    for i, column_x in enumerate(numeric_df.columns):
        for column_y in numeric_df.columns[i + 1:]:
            x = numeric_df[column_x].dropna()
            y = numeric_df[column_y].dropna()

            common_index = x.index.intersection(y.index)
            x_values = x.loc[common_index].to_numpy(dtype=float)
            y_values = y.loc[common_index].to_numpy(dtype=float)

            pearson_value = pearson_correlation(x_values, y_values)
            spearman_value = stats.spearmanr(
                x_values, y_values
            ).statistic

            rows.append(
                {
                    "Column_1": column_x,
                    "Column_2": column_y,
                    "Pearson": pearson_value,
                    "Spearman": spearman_value,
                }
            )

    return pd.DataFrame(rows)


def fit_normal_distribution(data: np.ndarray) -> dict:
    """Fit mean/std and perform a Shapiro-Wilk normality test."""
    data = np.asarray(data, dtype=float)

    mean = np.mean(data)
    std = np.std(data, ddof=0)

    shapiro_result = stats.shapiro(data)

    return {
        "mean": mean,
        "std": std,
        "shapiro_statistic": shapiro_result.statistic,
        "shapiro_p_value": shapiro_result.pvalue,
        "normally_distributed": shapiro_result.pvalue > 0.05,
    }


def compute_zscores(data: np.ndarray) -> np.ndarray:
    """Compute z-scores for the given data."""
    data = np.asarray(data, dtype=float)

    mean = np.mean(data)
    std = np.std(data, ddof=0)

    if std == 0:
        raise ValueError("Z-scores are undefined for constant data.")

    return (data - mean) / std