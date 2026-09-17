import pandas as pd
import numpy as np


# Series from a list
series_list = pd.Series([10, 20, 30, 40])
print("Series from list:")
print(series_list)


# Series from a dictionary
series_dict = pd.Series({"A": 100, "B": 200, "C": 300})
print("\nSeries from dictionary:")
print(series_dict)


# Series from NumPy array
array = np.array([50, 60, 70, 80])
series_array = pd.Series(array)
print("\nSeries from NumPy array:")
print(series_array)


# Series with custom index labels
custom_series = pd.Series(
    [90, 80, 70],
    index=["Math", "Science", "English"]
)
print("\nSeries with custom index labels:")
print(custom_series)


# DataFrame from dictionary of lists
df_dict = pd.DataFrame({
    "Name": ["Ravi", "Priya", "Anil"],
    "Age": [25, 28, 30],
    "Salary": [40000, 50000, 60000]
})

print("\nDataFrame from dictionary of lists:")
print(df_dict)


# DataFrame from list of dictionaries
df_list = pd.DataFrame([
    {"Name": "Ravi", "Age": 25, "Salary": 40000},
    {"Name": "Priya", "Age": 28, "Salary": 50000},
    {"Name": "Anil", "Age": 30, "Salary": 60000}
])

print("\nDataFrame from list of dictionaries:")
print(df_list)


# DataFrame from NumPy 2D array
array_2d = np.array([
    [1, 25, 40000],
    [2, 28, 50000],
    [3, 30, 60000]
])

df_numpy = pd.DataFrame(
    array_2d,
    columns=["ID", "Age", "Salary"]
)

print("\nDataFrame from NumPy 2D array:")
print(df_numpy)


# Create canonical dataset with 100 rows
employees = pd.DataFrame({
    "Employee_ID": range(1, 101),
    "Name": [f"Employee_{i}" for i in range(1, 101)],
    "Department": ["IT", "HR", "Finance", "Sales", "Marketing"] * 20,
    "Age": [25 + (i % 10) for i in range(100)],
    "Salary": [40000 + (i % 10) * 5000 for i in range(100)],
    "Joining_Date": pd.date_range("2020-01-01", periods=100, freq="D")
})

employees.to_csv("epic2_pandas/data/employees.csv", index=False)
employees.to_excel("epic2_pandas/data/employees.xlsx", index=False)
employees.to_json("epic2_pandas/data/employees.json", orient="records")

print("\nCanonical dataset created successfully.")
print(employees.head())
print("Shape:", employees.shape)


# Load data from CSV
csv_data = pd.read_csv("epic2_pandas/data/employees.csv")
print("\nCSV data loaded:")
print(csv_data.head())


# Load data from Excel
excel_data = pd.read_excel("epic2_pandas/data/employees.xlsx")
print("\nExcel data loaded:")
print(excel_data.head())


# Load data from JSON
json_data = pd.read_json("epic2_pandas/data/employees.json")
print("\nJSON data loaded:")
print(json_data.head())


# DataFrame methods
print("\nHead:")
print(csv_data.head())

print("\nTail:")
print(csv_data.tail())

print("\nInfo:")
csv_data.info()

print("\nDescribe:")
print(csv_data.describe())

print("\nData Types:")
print(csv_data.dtypes)

print("\nColumns:")
print(csv_data.columns)

print("\nShape:")
print(csv_data.shape)


# Function 1: Load dataset
def load_dataset(filepath: str) -> pd.DataFrame:
    if filepath.endswith(".csv"):
        return pd.read_csv(filepath)
    elif filepath.endswith(".xlsx"):
        return pd.read_excel(filepath)
    elif filepath.endswith(".json"):
        return pd.read_json(filepath)
    else:
        raise ValueError("Unsupported file format")


# Function 2: DataFrame summary
def dataframe_summary(df: pd.DataFrame) -> dict:
    return {
        "shape": df.shape,
        "dtypes": df.dtypes.to_dict(),
        "null_counts": df.isnull().sum().to_dict(),
        "memory_usage": df.memory_usage(deep=True).sum()
    }


# Function 3: Select columns and rows
def select_columns_rows(df: pd.DataFrame) -> None:
    print("\nColumn selection:")
    print(df["Name"])

    print("\nMultiple columns:")
    print(df[["Name", "Salary"]])

    print("\nUsing loc:")
    print(df.loc[0:4, ["Name", "Department"]])

    print("\nUsing iloc:")
    print(df.iloc[0:5, 0:3])

    print("\nRow slicing:")
    print(df[0:5])


# Test the 3 functions
csv_df = load_dataset("epic2_pandas/data/employees.csv")
excel_df = load_dataset("epic2_pandas/data/employees.xlsx")
json_df = load_dataset("epic2_pandas/data/employees.json")

print("\nLoaded CSV shape:", csv_df.shape)
print("Loaded Excel shape:", excel_df.shape)
print("Loaded JSON shape:", json_df.shape)

summary = dataframe_summary(csv_df)
print("\nDataFrame Summary:")
print(summary)

select_columns_rows(csv_df)