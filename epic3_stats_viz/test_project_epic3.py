from pathlib import Path

import numpy as np
import pandas as pd

from common.stats_utils import (
    compute_central_tendency,
    compute_spread,
    correlation_matrix_report,
    fit_normal_distribution,
    compute_zscores,
    pearson_correlation,
)

from common.viz_utils import (
    plot_histogram,
    plot_boxplot_by_category,
    plot_scatter_correlation,
)

from epic3_stats_viz.day11_descriptive_stats import (
    compute_central_tendency as original_central_tendency,
    compute_spread as original_spread,
)

from epic3_stats_viz.day12_correlation_distributions import (
    pearson_correlation as original_pearson_correlation,
    correlation_matrix_report as original_correlation_matrix_report,
    fit_normal_distribution as original_fit_normal_distribution,
    compute_zscores as original_compute_zscores,
)

from epic3_stats_viz.project_business_insights import (
    generate_full_report,
    compile_pdf_report,
)


DATASET_PATH = "epic2_pandas/data/cleaned_dataset.csv"


def test_dataset_loads():
    df = pd.read_csv(DATASET_PATH)

    assert not df.empty
    assert len(df) == 100


def test_descriptive_statistics():
    df = pd.read_csv(DATASET_PATH)

    data = df["Salary"].dropna().to_numpy()

    central = compute_central_tendency(data)
    spread = compute_spread(data)

    assert "mean" in central
    assert "median" in central
    assert "mode" in central
    assert "variance" in spread
    assert "std" in spread
    assert "range" in spread
    assert "iqr" in spread


def test_correlation_report():
    df = pd.read_csv(DATASET_PATH)

    report = correlation_matrix_report(df)

    assert not report.empty
    assert "Pearson" in report.columns
    assert "Spearman" in report.columns


def test_histogram_generation(tmp_path):
    df = pd.read_csv(DATASET_PATH)

    output = tmp_path / "histogram.png"

    plot_histogram(
        df["Age"],
        "Age",
        str(output),
    )

    assert output.exists()
    assert output.stat().st_size > 0


def test_boxplot_generation(tmp_path):
    df = pd.read_csv(DATASET_PATH)

    output = tmp_path / "boxplot.png"

    plot_boxplot_by_category(
        df,
        "Salary",
        "Department",
        str(output),
    )

    assert output.exists()
    assert output.stat().st_size > 0


def test_scatter_generation(tmp_path):
    df = pd.read_csv(DATASET_PATH)

    output = tmp_path / "scatter.png"

    plot_scatter_correlation(
        df,
        "Age",
        "Salary",
        str(output),
    )

    assert output.exists()
    assert output.stat().st_size > 0


def test_full_report_generation(tmp_path):
    df = pd.read_csv(DATASET_PATH)

    output = tmp_path / "business_report.pdf"

    generate_full_report(
        df,
        str(output),
    )

    assert output.exists()
    assert output.stat().st_size > 0


def test_compile_pdf_report(tmp_path):
    chart = tmp_path / "chart.png"

    df = pd.read_csv(DATASET_PATH)

    plot_histogram(
        df["Age"],
        "Age",
        str(chart),
    )

    output = tmp_path / "compiled_report.pdf"

    stats_summary = {
        "Age": {
            "Mean": 29.47,
            "Median": 29.36,
        }
    }

    compile_pdf_report(
        [chart],
        stats_summary,
        str(output),
    )

    assert output.exists()
    assert output.stat().st_size > 0


def test_generated_report_has_expected_name():
    report_path = Path(
        "epic3_stats_viz/reports/business_insights_report.pdf"
    )

    assert report_path.exists()
    assert report_path.stat().st_size > 0


def test_stats_utils_matches_day11():
    df = pd.read_csv(DATASET_PATH)
    data = df["Salary"].dropna().to_numpy()

    original_central = original_central_tendency(data)
    shared_central = compute_central_tendency(data)

    original_spread_result = original_spread(data)
    shared_spread = compute_spread(data)

    assert original_central == shared_central
    assert original_spread_result == shared_spread


def test_stats_utils_matches_day12_pearson():
    df = pd.read_csv(DATASET_PATH)

    x = df["Age"].to_numpy()
    y = df["Salary"].to_numpy()

    original_result = original_pearson_correlation(x, y)
    shared_result = pearson_correlation(x, y)

    assert np.isclose(
        original_result,
        shared_result,
    )


def test_stats_utils_matches_day12_correlation_report():
    df = pd.read_csv(DATASET_PATH)

    original_result = original_correlation_matrix_report(df)
    shared_result = correlation_matrix_report(df)

    pd.testing.assert_frame_equal(
        original_result,
        shared_result,
    )


def test_stats_utils_matches_day12_distribution_functions():
    df = pd.read_csv(DATASET_PATH)
    data = df["Age"].dropna().to_numpy()

    original_normal = original_fit_normal_distribution(data)
    shared_normal = fit_normal_distribution(data)

    assert original_normal == shared_normal

    original_zscore_result = original_compute_zscores(data)
    shared_zscore_result = compute_zscores(data)

    np.testing.assert_allclose(
        original_zscore_result,
        shared_zscore_result,
    )