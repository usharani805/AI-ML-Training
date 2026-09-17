import pytest
import pandas as pd
from day6_series_dataframes import load_dataset, dataframe_summary


DATA_PATH = "epic2_pandas/data/"


def test_load_csv():
    df = load_dataset(DATA_PATH + "employees.csv")
    assert isinstance(df, pd.DataFrame)
    assert df.shape == (100, 6)


def test_load_excel():
    df = load_dataset(DATA_PATH + "employees.xlsx")
    assert isinstance(df, pd.DataFrame)
    assert df.shape == (100, 6)


def test_load_json():
    df = load_dataset(DATA_PATH + "employees.json")
    assert isinstance(df, pd.DataFrame)
    assert df.shape == (100, 6)


def test_missing_file():
    with pytest.raises(FileNotFoundError):
        load_dataset(DATA_PATH + "missing.csv")


def test_malformed_file():
    with pytest.raises(ValueError):
        load_dataset(DATA_PATH + "employees.txt")


def test_dataframe_summary():
    df = load_dataset(DATA_PATH + "employees.csv")
    summary = dataframe_summary(df)

    assert summary["shape"] == (100, 6)
    assert "Employee_ID" in summary["dtypes"]
    assert summary["null_counts"]["Name"] == 0


def test_invalid_format():
    with pytest.raises(ValueError):
        load_dataset(DATA_PATH + "employees.pdf")