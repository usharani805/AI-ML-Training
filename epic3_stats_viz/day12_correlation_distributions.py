import numpy as np
import pandas as pd
from scipy import stats


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


if __name__ == "__main__":
    dataset_path = "epic2_pandas/data/cleaned_dataset.csv"
    df = pd.read_csv(dataset_path)

    print("\n=== Correlation Matrix Report ===")

    correlation_report = correlation_matrix_report(df)
    print(correlation_report.to_string(index=False))

    # Identify the strongest and weakest correlated pairs using
    # absolute Pearson correlation values.
    correlation_report["Abs_Pearson"] = (
        correlation_report["Pearson"].abs()
    )

    strongest_pairs = correlation_report.nlargest(
        2, "Abs_Pearson"
    )

    weakest_pairs = correlation_report.nsmallest(
        2, "Abs_Pearson"
    )

    print("\n=== 2 Strongest Correlated Pairs ===")
    print(
        strongest_pairs[
            ["Column_1", "Column_2", "Pearson", "Spearman"]
        ].to_string(index=False)
    )

    print("\n=== 2 Weakest Correlated Pairs ===")
    print(
        weakest_pairs[
            ["Column_1", "Column_2", "Pearson", "Spearman"]
        ].to_string(index=False)
    )

    # Report:
    # Strongest and weakest pairs are identified from the
    # actual Epic 2 cleaned dataset when this script runs.

    print("\n=== Normality Test Results ===")

    numeric_columns = df.select_dtypes(include=np.number).columns

    for column in numeric_columns[:2]:
        data = df[column].dropna().to_numpy(dtype=float)
        result = fit_normal_distribution(data)

        interpretation = (
            "Normally distributed"
            if result["normally_distributed"]
            else "Not normally distributed"
        )

        print(f"\nColumn: {column}")
        print(f"Mean: {result['mean']:.4f}")
        print(f"Standard deviation: {result['std']:.4f}")
        print(f"Shapiro statistic: {result['shapiro_statistic']:.4f}")
        print(f"Shapiro p-value: {result['shapiro_p_value']:.6f}")
        print(f"Interpretation: {interpretation}")

    print("\n=== Distribution Samples ===")

    np.random.seed(42)

    normal_samples = np.random.normal(
        loc=50, scale=10, size=1000
    )
    binomial_samples = np.random.binomial(
        n=10, p=0.5, size=1000
    )
    poisson_samples = np.random.poisson(
        lam=4, size=1000
    )

    print(
        f"Normal: mean={np.mean(normal_samples):.2f}, "
        f"std={np.std(normal_samples):.2f}"
    )
    print(
        f"Binomial: mean={np.mean(binomial_samples):.2f}, "
        f"std={np.std(binomial_samples):.2f}"
    )
    print(
        f"Poisson: mean={np.mean(poisson_samples):.2f}, "
        f"std={np.std(poisson_samples):.2f}"
    )

    print("\n=== Z-Score Outlier Detection ===")

    salary_data = df["Salary"].dropna().to_numpy(dtype=float)
    zscores = compute_zscores(salary_data)

    beyond_2_sd = salary_data[np.abs(zscores) > 2]
    beyond_3_sd = salary_data[np.abs(zscores) > 3]

    print(f"Values beyond ±2 SD: {beyond_2_sd}")
    print(f"Values beyond ±3 SD: {beyond_3_sd}")