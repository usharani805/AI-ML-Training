import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


sns.set_theme(style="whitegrid")


def plot_histogram(
    data: pd.Series,
    column_name: str,
    save_path: str,
) -> None:
    plt.figure(figsize=(8, 5))
    plt.hist(data.dropna(), bins=10)
    plt.title(f"{column_name} Distribution")
    plt.xlabel(column_name)
    plt.ylabel("Frequency")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_boxplot_by_category(
    df: pd.DataFrame,
    num_col: str,
    cat_col: str,
    save_path: str,
) -> None:
    plt.figure(figsize=(8, 5))

    categories = df[cat_col].dropna().unique()

    values = [
        df.loc[
            df[cat_col] == category,
            num_col,
        ].dropna()
        for category in categories
    ]

    plt.boxplot(
        values,
        tick_labels=categories,
    )

    plt.title(f"{num_col} by {cat_col}")
    plt.xlabel(cat_col)
    plt.ylabel(num_col)

    plt.savefig(
        save_path,
        dpi=150,
        bbox_inches="tight",
    )

    plt.close()


def plot_scatter_correlation(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    save_path: str,
) -> None:
    plt.figure(figsize=(8, 5))

    plt.scatter(
        df[x_col],
        df[y_col],
    )

    plt.title(f"{x_col} vs {y_col}")
    plt.xlabel(x_col)
    plt.ylabel(y_col)

    plt.savefig(
        save_path,
        dpi=150,
        bbox_inches="tight",
    )

    plt.close()


def plot_line_trend(
    df: pd.DataFrame,
    date_col: str,
    value_col: str,
    save_path: str,
) -> None:
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])

    trend = df.groupby(date_col)[value_col].mean()

    plt.figure(figsize=(8, 5))

    plt.plot(
        trend.index,
        trend.values,
        label=f"Average {value_col}",
    )

    plt.title(f"Average {value_col} Trend Over Time")
    plt.xlabel(date_col)
    plt.ylabel(f"Average {value_col}")
    plt.legend()

    plt.savefig(
        save_path,
        dpi=150,
        bbox_inches="tight",
    )

    plt.close()


def plot_dashboard_grid(
    df: pd.DataFrame,
    save_path: str,
) -> None:
    fig, axes = plt.subplots(
        2,
        2,
        figsize=(12, 9),
    )

    axes[0, 0].hist(
        df["Age"].dropna(),
        bins=10,
    )
    axes[0, 0].set_title("Age Distribution")
    axes[0, 0].set_xlabel("Age")
    axes[0, 0].set_ylabel("Frequency")

    categories = df["Department"].dropna().unique()

    values = [
        df.loc[
            df["Department"] == category,
            "Salary",
        ].dropna()
        for category in categories
    ]

    axes[0, 1].boxplot(
        values,
        tick_labels=categories,
    )
    axes[0, 1].set_title("Salary by Department")
    axes[0, 1].set_xlabel("Department")
    axes[0, 1].set_ylabel("Salary")

    axes[1, 0].scatter(
        df["Age"],
        df["Salary"],
    )
    axes[1, 0].set_title("Age vs Salary")
    axes[1, 0].set_xlabel("Age")
    axes[1, 0].set_ylabel("Salary")

    department_counts = df["Department"].value_counts()

    axes[1, 1].bar(
        department_counts.index,
        department_counts.values,
    )
    axes[1, 1].set_title("Employee Count by Department")
    axes[1, 1].set_xlabel("Department")
    axes[1, 1].set_ylabel("Employee Count")

    fig.suptitle("Employee Analytics Dashboard")
    fig.tight_layout()

    fig.savefig(
        save_path,
        dpi=150,
        bbox_inches="tight",
    )

    plt.close(fig)


def plot_correlation_heatmap(
    df: pd.DataFrame,
    save_path: str,
) -> None:
    correlation_matrix = (
        df.select_dtypes(include="number").corr()
    )

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

    plt.savefig(
        save_path,
        dpi=150,
        bbox_inches="tight",
    )

    plt.close()


def plot_pairplot(
    df: pd.DataFrame,
    hue_col: str,
    save_path: str,
) -> None:
    numeric_columns = (
        df.select_dtypes(include="number").columns
    )

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

    plt.title(
        f"{num_col} Distribution by {cat_col}"
    )
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

    plt.title(
        f"{x_col} vs {y_col} with Regression Line"
    )
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.tight_layout()

    plt.savefig(
        save_path,
        dpi=150,
        bbox_inches="tight",
    )

    plt.close()


def plot_department_countplot(
    df: pd.DataFrame,
    save_path: str,
) -> None:
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
        save_path,
        dpi=150,
        bbox_inches="tight",
    )

    plt.close()


def plot_matplotlib_vs_seaborn_histogram(
    df: pd.DataFrame,
    save_path: str,
) -> None:
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
        save_path,
        dpi=150,
        bbox_inches="tight",
    )

    plt.close(fig)
