
"""Validate StockNet sample generation across all stocks and phases."""

from pathlib import Path
import sys
import csv
import time

ROOT = Path(__file__).resolve().parent.parent

sys.path.insert(0, str(ROOT / "m2_workspace"))
sys.path.insert(0, str(ROOT / "m3_workspace"))

from stocknet_mappings import StockNetMappings
from price_reader import PriceReader
from tweet import TweetReader
from sample_builder import StockNetSampleBuilder


DATA = ROOT / "stocknet-dataset"
CODE = ROOT / "stocknet-code"

# Initialize mappings.
mappings = StockNetMappings(
    config_path=CODE / "src" / "config.yml",
    vocab_path=CODE / "res" / "vocab.txt",
)

# Initialize readers.
price_reader = PriceReader(
    DATA / "price" / "preprocessed"
)

tweet_reader = TweetReader(
    tweet_root=DATA / "tweet" / "preprocessed",
    vocab_path=CODE / "res" / "vocab.txt",
)

# Disable shuffling to make validation repeatable.
sample_builder = StockNetSampleBuilder(
    mappings=mappings,
    price_reader=price_reader,
    tweet_reader=tweet_reader,
    max_n_days=5,
    max_n_msgs=30,
    max_n_words=40,
    shuffle=False,
)

stocks = list(mappings.stock_id_dict.keys())
phases = ["train", "dev", "test"]

results = []

print("=" * 65)
print("STOCKNET FULL DATASET VALIDATION")
print("=" * 65)

print("Total stocks:", len(stocks))
print("Phases:", phases)

overall_start = time.time()

for phase in phases:

    print(f"\nVALIDATING PHASE: {phase.upper()}")
    print("-" * 65)

    phase_total = 0
    phase_errors = 0
    phase_start = time.time()

    for index, stock in enumerate(stocks, start=1):

        count = 0
        dates = []

        try:
            for sample in sample_builder.iter_samples(stock, phase):
                count += 1
                dates.append(sample["main_target_date"])

            status = "OK" if count > 0 else "NO_VALID_SAMPLES"

            results.append({
                "phase": phase,
                "stock": stock,
                "valid_samples": count,
                "earliest_date": min(dates) if dates else "",
                "latest_date": max(dates) if dates else "",
                "status": status,
                "error": "",
            })

        except Exception as exc:

            phase_errors += 1

            results.append({
                "phase": phase,
                "stock": stock,
                "valid_samples": count,
                "earliest_date": min(dates) if dates else "",
                "latest_date": max(dates) if dates else "",
                "status": "ERROR",
                "error": repr(exc),
            })

        phase_total += count

        print(
            f"[{index:02d}/{len(stocks)}] "
            f"{stock:8s} | "
            f"Samples: {count:5d} | "
            f"{results[-1]['status']}"
        )

    elapsed = time.time() - phase_start

    print("-" * 65)
    print("Phase:", phase)
    print("Total valid samples:", phase_total)
    print("Stocks with errors:", phase_errors)
    print("Time taken:", round(elapsed, 2), "seconds")

# Save detailed results.
report_path = ROOT / "m2_workspace" / "full_dataset_validation.csv"

with report_path.open("w", newline="", encoding="utf-8") as file:
    writer = csv.DictWriter(
        file,
        fieldnames=[
            "phase",
            "stock",
            "valid_samples",
            "earliest_date",
            "latest_date",
            "status",
            "error",
        ],
    )

    writer.writeheader()
    writer.writerows(results)

# Final summary.
print("\n" + "=" * 65)
print("FINAL VALIDATION SUMMARY")
print("=" * 65)

for phase in phases:

    phase_results = [
        row for row in results
        if row["phase"] == phase
    ]

    total = sum(row["valid_samples"] for row in phase_results)

    successful = sum(
        row["status"] == "OK"
        for row in phase_results
    )

    no_samples = sum(
        row["status"] == "NO_VALID_SAMPLES"
        for row in phase_results
    )

    errors = sum(
        row["status"] == "ERROR"
        for row in phase_results
    )

    print(f"\n{phase.upper()}")
    print("Valid samples:", total)
    print("Stocks with valid samples:", successful)
    print("Stocks with no valid samples:", no_samples)
    print("Stocks with errors:", errors)

print("\nReport saved to:", report_path)
print("Total execution time:", round(time.time() - overall_start, 2), "seconds")
