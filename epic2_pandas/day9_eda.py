import json
import pandas as pd


def univariate_report(df: pd.DataFrame) -> dict:
    """Generate univariate statistics for numeric and categorical columns."""
    report = {
        "numeric": {},
        "categorical": {}
    }

    numeric_columns = df.select_dtypes(include="number").columns
    categorical_columns = df.select_dtypes(include=["object", "str"]).columns

    for column in numeric_columns:
        report["numeric"][column] = {
            "count": int(df[column].count()),
            "mean": float(df[column].mean()),
            "median": float(df[column].median()),
            "min": float(df[column].min()),
            "max": float(df[column].max()),
            "std": float(df[column].std())
        }

    for column in categorical_columns:
        report["categorical"][column] = {
            str(key): int(value)
            for key, value in df[column].value_counts().items()
        }

    return report


def detect_outliers_iqr(
    df: pd.DataFrame, column: str
) -> pd.DataFrame:
    """Return rows where the selected numeric column has IQR outliers."""
    q1 = df[column].quantile(0.25)
    q3 = df[column].quantile(0.75)
    iqr = q3 - q1

    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    return df[
        (df[column] < lower_bound) |
        (df[column] > upper_bound)
    ]


def build_pivot_summary(
    df: pd.DataFrame,
    index: str,
    columns: str,
    values: str,
    aggfunc: str
) -> pd.DataFrame:
    """Build a pivot table summary."""
    return pd.pivot_table(
        df,
        index=index,
        columns=columns,
        values=values,
        aggfunc=aggfunc
    )


def categorize_column(
    df: pd.DataFrame,
    column: str,
    bins: list,
    labels: list
) -> pd.DataFrame:
    """Categorize a numeric column into bins."""
    result = df.copy()

    result[f"{column}_Category"] = pd.cut(
        result[column],
        bins=bins,
        labels=labels,
        include_lowest=True
    )

    return result


def generate_eda_report(df: pd.DataFrame) -> dict:
    """Generate the complete EDA report."""
    report = {}

    # Univariate statistics
    report["univariate_statistics"] = univariate_report(df)

    # Correlation matrix
    numeric_columns = df.select_dtypes(include="number").columns

    correlation = df[numeric_columns].corr().round(4)

    report["correlation_matrix"] = correlation.to_dict()

    # IQR outliers
    report["outliers"] = {}

    for column in numeric_columns:
        outliers = detect_outliers_iqr(df, column)

        report["outliers"][column] = {
            "count": int(len(outliers)),
            "rows": outliers.to_dict(orient="records")
        }

    # Pivot table 1
    pivot_salary = build_pivot_summary(
        df,
        index="Department",
        columns="Name",
        values="Salary",
        aggfunc="mean"
    )

    # Pivot table 2
    pivot_age = build_pivot_summary(
        df,
        index="Department",
        columns="Name",
        values="Age",
        aggfunc="mean"
    )

    report["pivot_tables"] = {
        "department_name_salary_mean":
            pivot_salary.fillna(0).to_dict(),
        "department_name_age_mean":
            pivot_age.fillna(0).to_dict()
    }

    # pd.cut categorization
    categorized_df = categorize_column(
        df,
        column="Salary",
        bins=[0, 50000, 60000, float("inf")],
        labels=["Low", "Medium", "High"]
    )

    report["salary_categories"] = {
        str(key): int(value)
        for key, value in
        categorized_df["Salary_Category"].value_counts().items()
    }

    # apply() transformation
    age_groups = df["Age"].apply(
        lambda age: "Below_30" if age < 30 else "30_and_above"
    )

    report["age_groups_apply"] = {
        str(key): int(value)
        for key, value in age_groups.value_counts().items()
    }

    # map() transformation
    department_map = {
        "IT": "Technology",
        "HR": "Human Resources",
        "Finance": "Finance",
        "Sales": "Sales",
        "Marketing": "Marketing"
    }

    mapped_departments = df["Department"].map(department_map)

    report["department_mapping_map"] = {
        str(key): int(value)
        for key, value in mapped_departments.value_counts().items()
    }

    # Data-driven insights
    department_salary = df.groupby("Department")["Salary"].mean()

    highest_salary_department = department_salary.idxmax()
    lowest_salary_department = department_salary.idxmin()

    highest_salary = float(department_salary.max())
    lowest_salary = float(department_salary.min())

    total_outliers = sum(
        item["count"] for item in report["outliers"].values()
    )

    correlation_values = correlation.abs().copy()

    for column in correlation_values.columns:
        correlation_values.loc[column, column] = 0

    strongest_pair = correlation_values.stack().idxmax()
    strongest_correlation = float(
        correlation.loc[strongest_pair[0], strongest_pair[1]]
    )

    report["insights"] = [
        (
            f"{highest_salary_department} has the highest average salary "
            f"at {highest_salary:.2f}."
        ),
        (
            f"{lowest_salary_department} has the lowest average salary "
            f"at {lowest_salary:.2f}."
        ),
        (
            f"The strongest numeric correlation is between "
            f"{strongest_pair[0]} and {strongest_pair[1]}, "
            f"with a correlation of {strongest_correlation:.3f}."
        ),
        (
            f"IQR analysis detected {total_outliers} outlier rows "
            f"across the numeric columns."
        )
    ]

    return report


if __name__ == "__main__":
    data_path = "epic2_pandas/data/cleaned_dataset.csv"
    output_path = "epic2_pandas/reports/eda_report.json"

    df = pd.read_csv(data_path)

    report = generate_eda_report(df)

    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(report, file, indent=4, default=str)

    print("EDA report generated successfully.")
    print(f"Report saved to: {output_path}")

    print("\nINSIGHTS:")
    for insight in report["insights"]:
        print("-", insight)