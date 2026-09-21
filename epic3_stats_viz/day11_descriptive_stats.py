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

    # Population variance and standard deviation use ddof=0.
    # Use ddof=0 when the data represents the complete population.
    #
    # Sample variance and standard deviation use ddof=1.
    # Use ddof=1 when the data represents a sample of a larger population.

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


if __name__ == "__main__":
    dataset_path = "epic2_pandas/data/cleaned_dataset.csv"
    df = pd.read_csv(dataset_path)

    numeric_columns = df.select_dtypes(include=np.number).columns

    print("\n=== Descriptive Statistics Comparison ===")

    for column in numeric_columns:
        data = df[column].dropna().to_numpy(dtype=float)

        central = compute_central_tendency(data)
        spread = compute_spread(data)

        library_results = {
            "mean": np.mean(data),
            "median": np.median(data),
            "mode": stats.mode(data, keepdims=False).mode,
            "variance": np.var(data, ddof=0),
            "std": np.std(data, ddof=0),
            "range": np.ptp(data),
            "iqr": stats.iqr(data),
        }

        comparison_table = pd.DataFrame(
            {
                "Statistic": [
                    "Mean",
                    "Median",
                    "Mode",
                    "Variance",
                    "Standard Deviation",
                    "Range",
                    "IQR",
                ],
                "From Scratch": [
                    central["mean"],
                    central["median"],
                    central["mode"],
                    spread["variance"],
                    spread["std"],
                    spread["range"],
                    spread["iqr"],
                ],
                "Library": [
                    library_results["mean"],
                    library_results["median"],
                    library_results["mode"],
                    library_results["variance"],
                    library_results["std"],
                    library_results["range"],
                    library_results["iqr"],
                ],
            }
        )

        print(f"\nColumn: {column}")
        print(comparison_table.to_string(index=False))

    # Probability scenario using the Epic 2 dataset.
    median_salary = df["Salary"].median()

    df["High_Salary"] = df["Salary"] > median_salary
    df["IT_Department"] = df["Department"] == "IT"

    probability_result = conditional_probability(
        df,
        "High_Salary",
        "IT_Department",
    )

    print("\n=== Probability Example ===")
    print(
        f"P(High Salary | IT Department) = "
        f"{probability_result:.6f}"
    )

    # Bayes' theorem worked example.
    # A = Spam email
    # B = Suspicious keyword is present
    p_a = 0.20
    p_b_given_a = 0.80
    p_b = 0.30

    bayes_result = bayes_theorem(
        p_a,
        p_b_given_a,
        p_b,
    )

    print("\n=== Bayes' Theorem Example ===")
    print(f"P(Spam) = {p_a}")
    print(f"P(Keyword | Spam) = {p_b_given_a}")
    print(f"P(Keyword) = {p_b}")
    print(f"P(Spam | Keyword) = {bayes_result:.6f}")
    print(
        "Interpretation: If an email contains the suspicious "
        "keyword, the probability that it is spam is "
        f"{bayes_result:.2%}."
    )