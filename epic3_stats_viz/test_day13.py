import pandas as pd

from epic3_stats_viz.day13_matplotlib_charts import (
    plot_boxplot_by_category,
    plot_dashboard_grid,
    plot_histogram,
    plot_scatter_correlation,
)


def test_plot_histogram(tmp_path):
    data = pd.Series([20, 25, 30, 35, 40])
    save_path = tmp_path / "histogram.png"

    plot_histogram(data, "Age", str(save_path))

    assert save_path.exists()
    assert save_path.stat().st_size > 0


def test_plot_histogram_empty_series(tmp_path):
    data = pd.Series([], dtype=float)
    save_path = tmp_path / "empty_histogram.png"

    plot_histogram(data, "Age", str(save_path))

    assert save_path.exists()
    assert save_path.stat().st_size > 0


def test_plot_boxplot_by_category(tmp_path):
    df = pd.DataFrame(
        {
            "Department": ["IT", "HR", "IT", "HR"],
            "Salary": [30000, 35000, 40000, 45000],
        }
    )
    save_path = tmp_path / "boxplot.png"

    plot_boxplot_by_category(df, "Salary", "Department", str(save_path))

    assert save_path.exists()
    assert save_path.stat().st_size > 0


def test_plot_boxplot_single_category(tmp_path):
    df = pd.DataFrame(
        {
            "Department": ["IT", "IT", "IT"],
            "Salary": [30000, 35000, 40000],
        }
    )
    save_path = tmp_path / "single_category_boxplot.png"

    plot_boxplot_by_category(df, "Salary", "Department", str(save_path))

    assert save_path.exists()
    assert save_path.stat().st_size > 0


def test_plot_scatter_correlation(tmp_path):
    df = pd.DataFrame(
        {
            "Age": [22, 25, 30, 35, 40],
            "Salary": [25000, 30000, 35000, 40000, 45000],
        }
    )
    save_path = tmp_path / "scatter.png"

    plot_scatter_correlation(df, "Age", "Salary", str(save_path))

    assert save_path.exists()
    assert save_path.stat().st_size > 0


def test_plot_dashboard_grid(tmp_path):
    df = pd.DataFrame(
        {
            "Age": [22, 25, 30, 35, 40],
            "Salary": [25000, 30000, 35000, 40000, 45000],
            "Department": ["IT", "HR", "IT", "Finance", "HR"],
        }
    )
    save_path = tmp_path / "dashboard.png"

    plot_dashboard_grid(df, str(save_path))

    assert save_path.exists()
    assert save_path.stat().st_size > 0