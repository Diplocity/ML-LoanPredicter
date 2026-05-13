import pandas as pd

# Load dataset
df = pd.read_csv("data/train_loandata.csv")

# Display first rows
print("FIRST 5 ROWS:")
print(df.head())

# Dataset information
print("\nDATASET INFO:")
print(df.info())

# Missing values
print("\nMISSING VALUES:")
print(df.isnull().sum())

# Statistics
print("\nSTATISTICS:")
print(df.describe())