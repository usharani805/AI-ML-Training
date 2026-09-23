import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sns.set_theme(style="whitegrid")


def plot_correlation_heatmap(df: pd.DataFrame, save_path: str) -> None:
    correlation_matrix = df.select_dtypes(include="number").corr()

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        correlation_matrix,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        linewidths=0.5,
    )
    plt.title("Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_pairplot(
    df: pd.DataFrame,
    hue_col: str,
    save_path: str,
) -> None:
    numeric_columns = df.select_dtypes(include="number").columns

    pairplot = sns.pairplot(
        df,
        vars=numeric_columns,
        hue=hue_col,
    )
    pairplot.fig.suptitle(
        "Pairplot of Numeric Features",
        y=1.02,
    )
    pairplot.savefig(
        save_path,
        dpi=150,
        bbox_inches="tight",
    )
    plt.close(pairplot.fig)


def plot_violin_by_category(
    df: pd.DataFrame,
    num_col: str,
    cat_col: str,
    save_path: str,
) -> None:
    plt.figure(figsize=(8, 5))

    sns.violinplot(
        data=df,
        x=cat_col,
        y=num_col,
    )

    plt.title(f"{num_col} Distribution by {cat_col}")
    plt.xlabel(cat_col)
    plt.ylabel(num_col)
    plt.tight_layout()

    plt.savefig(
        save_path,
        dpi=150,
        bbox_inches="tight",
    )

    plt.close()


def plot_regression_scatter(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    save_path: str,
) -> None:
    plt.figure(figsize=(8, 5))

    sns.regplot(
        data=df,
        x=x_col,
        y=y_col,
    )

    plt.title(f"{x_col} vs {y_col} with Regression Line")
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.tight_layout()

    plt.savefig(
        save_path,
        dpi=150,
        bbox_inches="tight",
    )

    plt.close()


if __name__ == "__main__":
    df = pd.read_csv(
        "epic2_pandas/data/cleaned_dataset.csv"
    )

    numeric_columns = df.select_dtypes(
        include="number"
    ).columns

    correlation_matrix = df[numeric_columns].corr()

    plot_correlation_heatmap(
        df,
        "epic3_stats_viz/charts/correlation_heatmap.png",
    )

    plot_pairplot(
        df,
        "Department",
        "epic3_stats_viz/charts/pairplot.png",
    )

    plot_violin_by_category(
        df,
        "Salary",
        "Department",
        "epic3_stats_viz/charts/salary_violin.png",
    )

    plot_regression_scatter(
        df,
        "Age",
        "Salary",
        "epic3_stats_viz/charts/age_salary_regression.png",
    )

    plt.figure(figsize=(8, 5))

    sns.countplot(
        data=df,
        x="Department",
    )

    plt.title("Employee Count by Department")
    plt.xlabel("Department")
    plt.ylabel("Employee Count")
    plt.tight_layout()

    plt.savefig(
        "epic3_stats_viz/charts/department_countplot.png",
        dpi=150,
        bbox_inches="tight",
    )

    plt.close()

    # Day 13 Matplotlib vs Day 14 Seaborn comparison
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(12, 5),
    )

    axes[0].hist(
        df["Age"].dropna(),
        bins=10,
    )

    axes[0].set_title(
        "Day 13 - Matplotlib Histogram"
    )
    axes[0].set_xlabel("Age")
    axes[0].set_ylabel("Frequency")

    sns.histplot(
        data=df,
        x="Age",
        bins=10,
        ax=axes[1],
    )

    axes[1].set_title(
        "Day 14 - Seaborn Histogram"
    )
    axes[1].set_xlabel("Age")
    axes[1].set_ylabel("Frequency")

    fig.suptitle(
        "Matplotlib vs Seaborn Histogram Comparison"
    )

    fig.tight_layout()

    fig.savefig(
        "epic3_stats_viz/charts/matplotlib_vs_seaborn_histogram.png",
        dpi=150,
        bbox_inches="tight",
    )

    plt.close(fig)

    # Seaborn provides a cleaner default statistical style and consistent theme.
    # Matplotlib provides basic plotting control, while Seaborn improves readability.