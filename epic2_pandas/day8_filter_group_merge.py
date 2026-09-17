import pandas as pd


# Load Day 7 cleaned dataset
df = pd.read_csv("epic2_pandas/data/cleaned_dataset.csv")


# ---------------------------------------------------------
# Filtering Operations
# ---------------------------------------------------------

# AND (&) - Select IT employees with salary above 50000
filter_and = df[(df["Department"] == "IT") & (df["Salary"] > 50000)]
print("AND Filtering:")
print(filter_and)


# OR (|) - Select employees from IT or HR
filter_or = df[(df["Department"] == "IT") | (df["Department"] == "HR")]
print("\nOR Filtering:")
print(filter_or)


# isin() - Select multiple departments
filter_isin = df[df["Department"].isin(["IT", "HR"])]
print("\nisin() Filtering:")
print(filter_isin)


# between() - Select employees with age between 25 and 30
filter_between = df[df["Age"].between(25, 30)]
print("\nbetween() Filtering:")
print(filter_between)


# query() - Select employees with salary greater than 50000
filter_query = df.query("Salary > 50000")
print("\nquery() Filtering:")
print(filter_query)


# ---------------------------------------------------------
# Grouping and Aggregation
# ---------------------------------------------------------

# Group by Department and calculate salary summary
grouped_summary = df.groupby("Department")["Salary"].agg(
    ["sum", "mean", "min", "max"]
)

print("\nDepartment-wise Salary Summary:")
print(grouped_summary)


# Multiple aggregations using agg()
agg_summary = df.groupby("Department").agg(
    Total_Salary=("Salary", "sum"),
    Average_Salary=("Salary", "mean"),
    Minimum_Salary=("Salary", "min"),
    Maximum_Salary=("Salary", "max")
)

print("\nMultiple Aggregations:")
print(agg_summary)


# Group-wise salary normalization using transform()
df["Salary_Normalized"] = df.groupby("Department")["Salary"].transform(
    lambda x: (x - x.mean()) / x.std()
)

print("\nGroup-wise Salary Normalization:")
print(df[["Department", "Salary", "Salary_Normalized"]].head())


# ---------------------------------------------------------
# Lookup Dataset and Merge Operations
# ---------------------------------------------------------

departments = pd.read_csv("epic2_pandas/data/departments.csv")

# merge: Use when combining datasets based on a common column.
# join: Use when combining DataFrames mainly based on indexes.
# concat: Use when stacking DataFrames vertically or horizontally.


# Inner Join
inner_merge = pd.merge(
    df,
    departments,
    on="Department",
    how="inner"
)

print("\nInner Join Row Count:", len(inner_merge))


# Left Join
left_merge = pd.merge(
    df,
    departments,
    on="Department",
    how="left"
)

print("Left Join Row Count:", len(left_merge))


# Right Join
right_merge = pd.merge(
    df,
    departments,
    on="Department",
    how="right"
)

print("Right Join Row Count:", len(right_merge))


# Outer Join
outer_merge = pd.merge(
    df,
    departments,
    on="Department",
    how="outer"
)

print("Outer Join Row Count:", len(outer_merge))


# ---------------------------------------------------------
# Concat Operations
# ---------------------------------------------------------

# Vertical concatenation - combine rows
vertical_concat = pd.concat(
    [df.head(5), df.tail(5)],
    axis=0
)

print("\nVertical Concatenation:")
print(vertical_concat)


# Horizontal concatenation - combine columns
horizontal_concat = pd.concat(
    [
        df[["Employee_ID", "Name"]],
        df[["Department", "Salary"]]
    ],
    axis=1
)

print("\nHorizontal Concatenation:")
print(horizontal_concat.head())


# ---------------------------------------------------------
# Required Practical Functions
# ---------------------------------------------------------

def filter_records(df: pd.DataFrame, conditions: dict) -> pd.DataFrame:
    result = df.copy()

    for column, condition in conditions.items():
        operator = condition[0]
        value = condition[1]

        if operator == ">":
            result = result[result[column] > value]
        elif operator == "<":
            result = result[result[column] < value]
        elif operator == ">=":
            result = result[result[column] >= value]
        elif operator == "<=":
            result = result[result[column] <= value]
        elif operator == "==":
            result = result[result[column] == value]
        elif operator == "isin":
            result = result[result[column].isin(value)]

    return result


def group_aggregate(
    df: pd.DataFrame,
    group_cols: list,
    agg_map: dict
) -> pd.DataFrame:
    return df.groupby(group_cols).agg(agg_map).reset_index()


def merge_datasets(
    df1: pd.DataFrame,
    df2: pd.DataFrame,
    on: str,
    how: str
) -> pd.DataFrame:
    return pd.merge(df1, df2, on=on, how=how)


def top_n_per_group(
    df: pd.DataFrame,
    group_col: str,
    sort_col: str,
    n: int
) -> pd.DataFrame:
    return (
        df.sort_values(sort_col, ascending=False)
        .groupby(group_col)
        .head(n)
        .reset_index(drop=True)
    )