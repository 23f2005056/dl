import pandas as pd
from pathlib import Path

root = Path(r"D:\DOWNLOADS\stocknet\stocknet-dataset")

raw_path = root / "price" / "raw" / "AAPL.csv"
processed_path = root / "price" / "preprocessed" / "AAPL.txt"

# Read raw data
raw = pd.read_csv(raw_path)
raw.columns = raw.columns.str.strip()

# Read preprocessed data
processed = pd.read_csv(
    processed_path,
    sep="\t",
    header=None,
    names=["Date", "Movement", "Open", "High", "Low", "Close", "Volume"]
)

raw["Date"] = pd.to_datetime(raw["Date"])
processed["Date"] = pd.to_datetime(processed["Date"])

# Sort chronologically
raw = raw.sort_values("Date").reset_index(drop=True)
processed = processed.sort_values("Date").reset_index(drop=True)

# Previous day's adjusted close
raw["Previous_Adj_Close"] = raw["Adj Close"].shift(1)

# Candidate normalized values
raw["Expected_High"] = (
    raw["High"] / raw["Previous_Adj_Close"] - 1
)
raw["Expected_Low"] = (
    raw["Low"] / raw["Previous_Adj_Close"] - 1
)
raw["Expected_Close"] = (
    raw["Adj Close"] / raw["Previous_Adj_Close"] - 1
)

# Merge by date
merged = processed.merge(
    raw[
        [
            "Date",
            "Open",
            "High",
            "Low",
            "Close",
            "Adj Close",
            "Previous_Adj_Close",
            "Expected_High",
            "Expected_Low",
            "Expected_Close"
        ]
    ],
    on="Date",
    how="inner",
    suffixes=("_processed", "_raw")
)

# Display a small sample
print("\nMATCHED ROWS:", len(merged))

print("\nRAW-BASED EXPECTED VALUES:")
print(
    merged[
        [
            "Date",
            "High_raw",
            "Low_raw",
            "Close_raw",
            "Expected_High",
            "Expected_Low",
            "Expected_Close"
        ]
    ].tail(10).to_string(index=False)
)

print("\nPREPROCESSED VALUES:")
print(
    merged[
        [
            "Date",
            "High_processed",
            "Low_processed",
            "Close_processed"
        ]
    ].tail(10).to_string(index=False)
)
