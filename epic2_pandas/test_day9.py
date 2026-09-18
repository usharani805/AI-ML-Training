import pandas as pd

from day9_eda import (
    univariate_report,
    detect_outliers_iqr,
    build_pivot_summary,
    categorize_column,
    generate_eda_report,
)


def sample_df():
    return pd.DataFrame({
        "Name": ["A", "B", "C", "D", "E"],
        "Department": ["IT", "IT", "HR", "HR", "Sales"],
        "Age": [25, 30, 35, 40, 45],
        "Salary": [40000, 50000, 60000, 70000, 200000],
    })


def test_univariate_numeric_stats():
    df = sample_df()
    report = univariate_report(df)

    assert report["numeric"]["Salary"]["count"] == 5
    assert report["numeric"]["Salary"]["min"] == 40000.0
    assert report["numeric"]["Salary"]["max"] == 200000.0


def test_categorical_value_counts():
    df = sample_df()
    report = univariate_report(df)

    assert report["categorical"]["Department"]["IT"] == 2
    assert report["categorical"]["Department"]["HR"] == 2


def test_iqr_detects_known_outlier():
    df = pd.DataFrame({
        "Salary": [100, 110, 120, 130, 1000]
    })

    outliers = detect_outliers_iqr(df, "Salary")

    assert len(outliers) == 1
    assert outliers.iloc[0]["Salary"] == 1000


def test_iqr_no_outlier():
    df = pd.DataFrame({
        "Salary": [100, 110, 120, 130, 140]
    })

    outliers = detect_outliers_iqr(df, "Salary")

    assert outliers.empty


def test_pivot_shape():
    df = sample_df()

    pivot = build_pivot_summary(
        df,
        index="Department",
        columns="Name",
        values="Salary",
        aggfunc="mean"
    )

    assert pivot.shape == (3, 5)


def test_pivot_value():
    df = sample_df()

    pivot = build_pivot_summary(
        df,
        index="Department",
        columns="Name",
        values="Salary",
        aggfunc="mean"
    )

    assert pivot.loc["IT", "A"] == 40000


def test_categorize_bins():
    df = pd.DataFrame({
        "Salary": [30000, 50000, 60000, 80000]
    })

    result = categorize_column(
        df,
        "Salary",
        bins=[0, 50000, 60000, float("inf")],
        labels=["Low", "Medium", "High"]
    )

    assert str(result.loc[0, "Salary_Category"]) == "Low"
    assert str(result.loc[1, "Salary_Category"]) == "Low"
    assert str(result.loc[2, "Salary_Category"]) == "Medium"
    assert str(result.loc[3, "Salary_Category"]) == "High"


def test_categorize_preserves_columns():
    df = pd.DataFrame({
        "Salary": [30000, 80000]
    })

    result = categorize_column(
        df,
        "Salary",
        bins=[0, 50000, 60000, float("inf")],
        labels=["Low", "Medium", "High"]
    )

    assert "Salary" in result.columns
    assert "Salary_Category" in result.columns


def test_generate_report_structure():
    df = sample_df()

    report = generate_eda_report(df)

    assert "univariate_statistics" in report
    assert "correlation_matrix" in report
    assert "outliers" in report
    assert "pivot_tables" in report
    assert "salary_categories" in report
    assert "age_groups_apply" in report
    assert "department_mapping_map" in report
    assert "insights" in report

    assert len(report["insights"]) >= 3