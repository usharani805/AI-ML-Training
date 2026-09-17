import pandas as pd


def inject_dirty_data(
    df: pd.DataFrame,
    missing_pct: float,
    dup_count: int
) -> pd.DataFrame:
    dirty_df = df.copy()

    # Inject missing values
    total_cells = dirty_df.size
    missing_count = int(total_cells * missing_pct / 100)

    if missing_count > 0:
        for i in range(missing_count):
            row = i % len(dirty_df)
            col = i % len(dirty_df.columns)
            dirty_df.iloc[row, col] = pd.NA

    # Inject duplicate rows
    if dup_count > 0:
        duplicates = dirty_df.iloc[:dup_count].copy()
        dirty_df = pd.concat(
            [dirty_df, duplicates],
            ignore_index=True
        )

    return dirty_df


def handle_missing_values(
    df: pd.DataFrame,
    strategy: dict
) -> pd.DataFrame:
    cleaned_df = df.copy()

    for column, method in strategy.items():
        if column not in cleaned_df.columns:
            continue

        if method == "mean":
            cleaned_df[column] = cleaned_df[column].fillna(
                cleaned_df[column].mean()
            )

        elif method == "median":
            cleaned_df[column] = cleaned_df[column].fillna(
                cleaned_df[column].median()
            )

        elif method == "mode":
            mode_value = cleaned_df[column].mode()
            if not mode_value.empty:
                cleaned_df[column] = cleaned_df[column].fillna(
                    mode_value.iloc[0]
                )

        elif method == "drop":
            cleaned_df = cleaned_df.dropna(subset=[column])

        elif method == "ffill":
            cleaned_df[column] = cleaned_df[column].ffill()

        elif method == "bfill":
            cleaned_df[column] = cleaned_df[column].bfill()

    return cleaned_df


def remove_duplicates(
    df: pd.DataFrame,
    subset: list = None
) -> pd.DataFrame:
    cleaned_df = df.copy()

    # Detect duplicates
    duplicate_count = cleaned_df.duplicated(subset=subset).sum()
    print(f"Duplicate rows found: {duplicate_count}")

    # Remove duplicates
    cleaned_df = cleaned_df.drop_duplicates(subset=subset)

    return cleaned_df


def fix_dtypes(
    df: pd.DataFrame,
    dtype_map: dict
) -> pd.DataFrame:
    cleaned_df = df.copy()

    for column, dtype in dtype_map.items():
        if column not in cleaned_df.columns:
            continue

        if dtype == "datetime64":
            cleaned_df[column] = pd.to_datetime(
                cleaned_df[column],
                errors="coerce"
            )
        else:
            cleaned_df[column] = pd.to_numeric(
                cleaned_df[column],
                errors="coerce"
            )

    return cleaned_df


def data_quality_report(df: pd.DataFrame) -> dict:
    return {
        "shape": df.shape,
        "null_counts": df.isnull().sum().to_dict(),
        "duplicate_count": int(df.duplicated().sum()),
        "dtypes": df.dtypes.astype(str).to_dict(),
        "memory_usage": int(df.memory_usage(deep=True).sum())
    }


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    dirty_df = inject_dirty_data(
        df,
        missing_pct=2,
        dup_count=5
    )

    print("Before Cleaning:")
    print(data_quality_report(dirty_df))

    strategy = {
        "Employee_ID": "bfill",
        "Name": "mode",
        "Department": "mode",
        "Age": "mean",
        "Salary": "mean",
        "Joining_Date": "ffill"
    }

    cleaned_df = handle_missing_values(
        dirty_df,
        strategy
    )

    cleaned_df = remove_duplicates(cleaned_df)

    dtype_map = {
        "Employee_ID": "numeric",
        "Age": "numeric",
        "Salary": "numeric",
        "Joining_Date": "datetime64"
    }

    cleaned_df = fix_dtypes(
        cleaned_df,
        dtype_map
    )

    print("\nAfter Cleaning:")
    print(data_quality_report(cleaned_df))

    return cleaned_df


if __name__ == "__main__":
    df = pd.read_csv(
        "epic2_pandas/data/employees.csv"
    )

    cleaned_df = clean_dataset(df)

    cleaned_df.to_csv(
        "epic2_pandas/data/cleaned_dataset.csv",
        index=False
    )

    print("\nCleaned dataset saved successfully.")