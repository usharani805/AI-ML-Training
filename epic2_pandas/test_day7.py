import pandas as pd

from day7_data_cleaning import (
    handle_missing_values,
    remove_duplicates,
    fix_dtypes,
    data_quality_report,
    clean_dataset
)


def test_handle_missing_values_mean():
    df = pd.DataFrame({
        "Age": [20, None, 30]
    })

    result = handle_missing_values(
        df,
        {"Age": "mean"}
    )

    assert result["Age"].isnull().sum() == 0


def test_handle_missing_values_median():
    df = pd.DataFrame({
        "Age": [20, None, 40]
    })

    result = handle_missing_values(
        df,
        {"Age": "median"}
    )

    assert result["Age"].isnull().sum() == 0


def test_handle_missing_values_mode():
    df = pd.DataFrame({
        "Department": ["IT", "HR", None, "IT"]
    })

    result = handle_missing_values(
        df,
        {"Department": "mode"}
    )

    assert result["Department"].isnull().sum() == 0


def test_handle_missing_values_ffill():
    df = pd.DataFrame({
        "Joining_Date": ["2024-01-01", None, "2024-01-03"]
    })

    result = handle_missing_values(
        df,
        {"Joining_Date": "ffill"}
    )

    assert result["Joining_Date"].isnull().sum() == 0


def test_remove_duplicates():
    df = pd.DataFrame({
        "Name": ["Ravi", "Ravi", "Sita"],
        "Age": [25, 25, 30]
    })

    result = remove_duplicates(df)

    assert len(result) == 2
    assert result.duplicated().sum() == 0


def test_remove_duplicates_subset():
    df = pd.DataFrame({
        "Name": ["Ravi", "Ravi", "Sita"],
        "Age": [25, 30, 30]
    })

    result = remove_duplicates(
        df,
        subset=["Name"]
    )

    assert len(result) == 2


def test_fix_dtypes():
    df = pd.DataFrame({
        "Salary": ["50000", "60000"],
        "Joining_Date": ["2024-01-01", "2024-02-01"]
    })

    result = fix_dtypes(
        df,
        {
            "Salary": "numeric",
            "Joining_Date": "datetime64"
        }
    )

    assert pd.api.types.is_numeric_dtype(result["Salary"])
    assert pd.api.types.is_datetime64_any_dtype(
        result["Joining_Date"]
    )


def test_data_quality_report():
    df = pd.DataFrame({
        "Name": ["Ravi", None, "Sita"],
        "Age": [25, 30, 30]
    })

    report = data_quality_report(df)

    assert report["null_counts"]["Name"] == 1
    assert report["duplicate_count"] == 0


def test_clean_dataset():
    df = pd.DataFrame({
        "Employee_ID": range(1, 11),
        "Name": ["Ravi"] * 10,
        "Department": ["IT"] * 10,
        "Age": [25] * 10,
        "Salary": [50000] * 10,
        "Joining_Date": ["2024-01-01"] * 10
    })

    result = clean_dataset(df)

    assert result.isnull().sum().sum() == 0
    assert result.duplicated().sum() == 0


def test_empty_dataframe():
    df = pd.DataFrame(
        columns=[
            "Employee_ID",
            "Name",
            "Department",
            "Age",
            "Salary",
            "Joining_Date"
        ]
    )

    report = data_quality_report(df)

    assert report["shape"] == (0, 6)


def test_100_percent_missing_column():
    df = pd.DataFrame({
        "Age": [None, None, None]
    })

    result = handle_missing_values(
        df,
        {"Age": "mean"}
    )

    assert result["Age"].isnull().sum() == 3