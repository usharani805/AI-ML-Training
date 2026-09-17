import pandas as pd

from day8_filter_group_merge import (
    filter_records,
    group_aggregate,
    merge_datasets,
    top_n_per_group,
)


def sample_data():
    return pd.DataFrame({
        "Employee_ID": [1, 2, 3, 4],
        "Department": ["IT", "IT", "HR", "HR"],
        "Salary": [40000, 60000, 50000, 70000],
    })


def test_filter_records():
    df = sample_data()
    result = filter_records(df, {"Salary": (">", 50000)})
    assert len(result) == 2


def test_group_aggregate():
    df = sample_data()
    result = group_aggregate(
        df,
        ["Department"],
        {"Salary": ["sum", "mean"]}
    )

    it_row = result[result["Department"] == "IT"].iloc[0]

    assert it_row["Salary"]["sum"] == 100000
    assert it_row["Salary"]["mean"] == 50000


def test_inner_join():
    df1 = pd.DataFrame({"Department": ["IT", "HR", "Sales"]})
    df2 = pd.DataFrame({"Department": ["IT", "HR"]})

    result = merge_datasets(df1, df2, "Department", "inner")

    assert len(result) == 2


def test_left_join():
    df1 = pd.DataFrame({"Department": ["IT", "HR", "Sales"]})
    df2 = pd.DataFrame({"Department": ["IT", "HR"]})

    result = merge_datasets(df1, df2, "Department", "left")

    assert len(result) == 3


def test_right_join():
    df1 = pd.DataFrame({"Department": ["IT", "HR"]})
    df2 = pd.DataFrame({"Department": ["IT", "HR", "Finance"]})

    result = merge_datasets(df1, df2, "Department", "right")

    assert len(result) == 3


def test_outer_join():
    df1 = pd.DataFrame({"Department": ["IT", "HR"]})
    df2 = pd.DataFrame({"Department": ["HR", "Finance"]})

    result = merge_datasets(df1, df2, "Department", "outer")

    assert len(result) == 3


def test_top_n_per_group():
    df = sample_data()

    result = top_n_per_group(
        df,
        "Department",
        "Salary",
        1
    )

    assert len(result) == 2
    assert set(result["Salary"]) == {60000, 70000}