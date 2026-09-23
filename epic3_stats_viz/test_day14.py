from pathlib import Path

import numpy as np
import pandas as pd

from day14_seaborn_advanced import (
    plot_correlation_heatmap,
    plot_pairplot,
    plot_regression_scatter,
    plot_violin_by_category,
)


def create_test_dataframe():
    return pd.DataFrame(
        {
            "Age": [25, 30, 35, 40, 45, 50],
            "Salary": [30000, 40000, 50000, 60000, 70000, 80000],
            "Department": [
                "IT",
                "HR",
                "IT",
                "Finance",
                "HR",
                "Finance",
            ],
        }
    )


def test_heatmap_creates_file(tmp_path):
    df = create_test_dataframe()
    save_path = tmp_path / "heatmap.png"

    plot_correlation_heatmap(df, str(save_path))

    assert save_path.exists()
    assert save_path.stat().st_size > 0


def test_heatmap_values_match_correlation_matrix(tmp_path):
    df = create_test_dataframe()
    correlation_matrix = df.select_dtypes(include="number").corr()
    expected = np.array(correlation_matrix)

    save_path = tmp_path / "heatmap.png"
    plot_correlation_heatmap(df, str(save_path))

    assert np.allclose(correlation_matrix, expected)


def test_pairplot_creates_file(tmp_path):
    df = create_test_dataframe()
    save_path = tmp_path / "pairplot.png"

    plot_pairplot(df, "Department", str(save_path))

    assert save_path.exists()
    assert save_path.stat().st_size > 0


def test_violin_plot_creates_file(tmp_path):
    df = create_test_dataframe()
    save_path = tmp_path / "violin.png"

    plot_violin_by_category(
        df,
        "Salary",
        "Department",
        str(save_path),
    )

    assert save_path.exists()
    assert save_path.stat().st_size > 0


def test_regression_plot_creates_file(tmp_path):
    df = create_test_dataframe()
    save_path = tmp_path / "regression.png"

    plot_regression_scatter(
        df,
        "Age",
        "Salary",
        str(save_path),
    )

    assert save_path.exists()
    assert save_path.stat().st_size > 0


def test_all_required_columns_exist():
    df = create_test_dataframe()

    assert "Age" in df.columns
    assert "Salary" in df.columns
    assert "Department" in df.columns