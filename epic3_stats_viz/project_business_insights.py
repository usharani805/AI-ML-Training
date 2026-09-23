from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages

from common.stats_utils import (
    compute_central_tendency,
    compute_spread,
    correlation_matrix_report,
    fit_normal_distribution,
)

from common.viz_utils import (
    plot_histogram,
    plot_boxplot_by_category,
    plot_scatter_correlation,
    plot_line_trend,
    plot_dashboard_grid,
    plot_correlation_heatmap,
    plot_pairplot,
    plot_violin_by_category,
    plot_regression_scatter,
    plot_department_countplot,
    plot_matplotlib_vs_seaborn_histogram,
)


DATASET_PATH = "epic2_pandas/data/cleaned_dataset.csv"
REPORT_DIR = Path("epic3_stats_viz/reports")
CHART_DIR = Path("epic3_stats_viz/charts")
OUTPUT_PDF = REPORT_DIR / "business_insights_report.pdf"


def compile_pdf_report(
    chart_paths: list,
    stats_summary: dict,
    output_pdf: str,
) -> None:
    with PdfPages(output_pdf) as pdf:
        fig = plt.figure(figsize=(11, 8.5))
        fig.text(
            0.5,
            0.9,
            "Business Insights Report",
            ha="center",
            fontsize=22,
            fontweight="bold",
        )

        y_position = 0.82

        for column, statistics in stats_summary.items():
            fig.text(
                0.08,
                y_position,
                f"{column}",
                fontsize=14,
                fontweight="bold",
            )
            y_position -= 0.04

            for name, value in statistics.items():
                fig.text(
                    0.10,
                    y_position,
                    f"{name}: {value:.4f}",
                    fontsize=11,
                )
                y_position -= 0.035

            y_position -= 0.02

            if y_position < 0.12:
                pdf.savefig(fig)
                plt.close(fig)

                fig = plt.figure(figsize=(11, 8.5))
                y_position = 0.90

        pdf.savefig(fig)
        plt.close(fig)

        for chart_path in chart_paths:
            image = plt.imread(chart_path)

            fig = plt.figure(figsize=(11, 8.5))
            plt.imshow(image)
            plt.axis("off")
            plt.tight_layout()

            pdf.savefig(fig)
            plt.close(fig)


def generate_full_report(
    df: pd.DataFrame,
    output_path: str,
) -> None:
    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    CHART_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    df = df.copy()

    df["Joining_Date"] = pd.to_datetime(
        df["Joining_Date"]
    )

    numeric_columns = df.select_dtypes(
        include="number"
    ).columns

    stats_summary = {}

    for column in numeric_columns:
        data = (
            df[column]
            .dropna()
            .to_numpy(dtype=float)
        )

        central = compute_central_tendency(data)
        spread = compute_spread(data)

        stats_summary[column] = {
            "Mean": central["mean"],
            "Median": central["median"],
            "Mode": central["mode"],
            "Variance": spread["variance"],
            "Standard Deviation": spread["std"],
            "Range": spread["range"],
            "IQR": spread["iqr"],
        }

    correlation_report = correlation_matrix_report(df)

    stats_summary["Correlation Summary"] = {
        "Strongest Pearson Correlation": correlation_report[
            "Pearson"
        ].abs().max(),
        "Average Pearson Correlation": correlation_report[
            "Pearson"
        ].mean(),
    }

    for column in numeric_columns[:2]:
        data = (
            df[column]
            .dropna()
            .to_numpy(dtype=float)
        )

        normality_result = fit_normal_distribution(data)

        stats_summary[f"{column} Normality"] = {
            "Shapiro Statistic": normality_result[
                "shapiro_statistic"
            ],
            "Shapiro P-Value": normality_result[
                "shapiro_p_value"
            ],
        }

    chart_paths = [
        CHART_DIR / "project_age_histogram.png",
        CHART_DIR / "project_salary_boxplot.png",
        CHART_DIR / "project_age_salary_scatter.png",
        CHART_DIR / "project_salary_trend.png",
        CHART_DIR / "project_dashboard.png",
        CHART_DIR / "project_correlation_heatmap.png",
        CHART_DIR / "project_pairplot.png",
        CHART_DIR / "project_salary_violin.png",
        CHART_DIR / "project_age_salary_regression.png",
        CHART_DIR / "project_department_countplot.png",
        CHART_DIR / "project_matplotlib_vs_seaborn_histogram.png",
    ]

    plot_histogram(
        df["Age"],
        "Age",
        str(chart_paths[0]),
    )

    plot_boxplot_by_category(
        df,
        "Salary",
        "Department",
        str(chart_paths[1]),
    )

    plot_scatter_correlation(
        df,
        "Age",
        "Salary",
        str(chart_paths[2]),
    )

    plot_line_trend(
        df,
        "Joining_Date",
        "Salary",
        str(chart_paths[3]),
    )

    plot_dashboard_grid(
        df,
        str(chart_paths[4]),
    )

    plot_correlation_heatmap(
        df,
        str(chart_paths[5]),
    )

    plot_pairplot(
        df,
        "Department",
        str(chart_paths[6]),
    )

    plot_violin_by_category(
        df,
        "Salary",
        "Department",
        str(chart_paths[7]),
    )

    plot_regression_scatter(
        df,
        "Age",
        "Salary",
        str(chart_paths[8]),
    )

    plot_department_countplot(
        df,
        str(chart_paths[9]),
    )

    plot_matplotlib_vs_seaborn_histogram(
        df,
        str(chart_paths[10]),
    )

    compile_pdf_report(
        chart_paths,
        stats_summary,
        output_path,
    )


if __name__ == "__main__":
    data = pd.read_csv(DATASET_PATH)

    generate_full_report(
        data,
        str(OUTPUT_PDF),
    )

    print(
        f"Business Insights Report generated: "
        f"{OUTPUT_PDF}"
    )
