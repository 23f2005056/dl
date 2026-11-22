import os
from pathlib import Path
from datetime import datetime

PRICE_DIR = Path(
    r"D:\DOWNLOADS\stocknet\stocknet-dataset\price\preprocessed"
)

files = sorted(PRICE_DIR.glob("*.txt"))

print("=" * 75)
print("STOCKNET PRICE DATA AUDIT")
print("=" * 75)

print(f"Total price files: {len(files)}")
print("Expected files: 88")
print(f"File count matches: {len(files) == 88}")

print("\n--- First 3 lines of AAPL ---")

aapl_path = PRICE_DIR / "AAPL.txt"

if aapl_path.exists():
    with aapl_path.open(encoding="utf-8") as f:
        for _ in range(3):
            line = f.readline()
            if line:
                print(repr(line))
else:
    print("AAPL.txt NOT FOUND")

print("\n--- Individual stock audit ---")

header = (
    f"{'Stock':8} {'Rows':>7} {'Columns':>10} "
    f"{'Newest':12} {'Oldest':12} "
    f"{'Descending':>11} {'MV=Close':>10} {'Bad':>6}"
)

print(header)
print("-" * len(header))

problems = []

for path in files:

    stock = path.stem
    dates = []
    records = []
    bad = 0

    with path.open(encoding="utf-8") as f:

        for line in f:

            parts = line.split()

            if not parts:
                continue

            try:
                date = datetime.strptime(parts[0], "%Y-%m-%d")
                values = [float(x) for x in parts[1:]]

                dates.append(date)
                records.append(values)

            except (ValueError, IndexError):
                bad += 1

    descending = (
        all(dates[i] > dates[i + 1]
            for i in range(len(dates) - 1))
        if dates else False
    )

    column_counts = sorted(set(len(r) for r in records))

    mv_close_match = (
        sum(abs(r[0] - r[4]) < 1e-6 for r in records)
        / len(records)
        if records and all(len(r) >= 5 for r in records)
        else float("nan")
    )

    newest = dates[0].strftime("%Y-%m-%d") if dates else "N/A"
    oldest = dates[-1].strftime("%Y-%m-%d") if dates else "N/A"

    print(
        f"{stock:8} {len(records):7} {str(column_counts):>10} "
        f"{newest:12} {oldest:12} "
        f"{str(descending):>11} {mv_close_match:10.2f} {bad:6}"
    )

    duplicate_dates = len(dates) != len(set(dates))

    if (
        not dates
        or not descending
        or bad
        or duplicate_dates
        or len(column_counts) != 1
    ):
        problems.append(stock)

print("\n--- AUDIT SUMMARY ---")

print(f"Total files: {len(files)}")
print(f"Files with problems: {len(problems)}")
print("Problematic stocks:", problems)

print("\nAudit completed.")
