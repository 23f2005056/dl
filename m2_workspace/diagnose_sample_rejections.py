
"""Diagnose why StockNet target dates are rejected."""

from pathlib import Path
from datetime import timedelta
from collections import Counter
import sys
import csv
import json
import time

ROOT = Path(__file__).resolve().parent.parent

sys.path.insert(0, str(ROOT / "m2_workspace"))
sys.path.insert(0, str(ROOT / "m3_workspace"))

from stocknet_mappings import StockNetMappings
from price_reader import PriceReader
from tweet import TweetReader
from sample_builder import StockNetSampleBuilder
from date_alignment import build_price_timeline


DATA = ROOT / "stocknet-dataset"
CODE = ROOT / "stocknet-code"

mappings = StockNetMappings(
    config_path=CODE / "src" / "config.yml",
    vocab_path=CODE / "res" / "vocab.txt",
)

price_reader = PriceReader(DATA / "price" / "preprocessed")

tweet_reader = TweetReader(
    tweet_root=DATA / "tweet" / "preprocessed",
    vocab_path=CODE / "res" / "vocab.txt",
)

builder = StockNetSampleBuilder(
    mappings=mappings,
    price_reader=price_reader,
    tweet_reader=tweet_reader,
    max_n_days=5,
    max_n_msgs=30,
    max_n_words=40,
    shuffle=False,
)

PHASES = ["train", "dev", "test"]

details = []
summary = []
overall_start = time.time()


def diagnose_price_timeline(records, target_date):
    """Identify the reason build_price_timeline rejects a target."""

    if target_date not in records:
        return "PRICE_DATE_MISSING", None

    movement = records[target_date]["movement"]

    if -0.005 <= movement < 0.0055:
        return "PRICE_MOVEMENT_FILTER", None

    d_t_min = target_date - timedelta(days=4)

    all_dates = sorted(records)

    ts = [
        d for d in all_dates
        if d_t_min <= d <= target_date
    ]

    if not ts or ts[-1] != target_date:
        return "PRICE_TIMELINE_INVALID", None

    earlier_dates = [d for d in all_dates if d < d_t_min]

    if not earlier_dates:
        return "PRICE_TIMELINE_INVALID", None

    price_dates = [earlier_dates[-1]] + [
        d for d in ts if d < target_date
    ]

    if len(price_dates) != len(ts):
        return "PRICE_TIMELINE_INVALID", None

    price_info = build_price_timeline(
        records,
        target_date,
        max_n_days=5,
    )

    if price_info is None:
        return "PRICE_TIMELINE_INVALID", None

    return "PRICE_OK", price_info


def diagnose_tweet_alignment(stock, target_date, ts):
    """Count messages assigned to each trading step."""

    counts = [0] * len(ts)
    tweet_file_counts = [0] * len(ts)

    tweet_dates = [
        target_date - timedelta(days=offset)
        for offset in range(5, 0, -1)
    ]

    for tweet_date in tweet_dates:
        record = tweet_reader.get(stock, tweet_date.isoformat())

        step_index = next(
            (
                i for i, trading_date in enumerate(ts)
                if tweet_date < trading_date
            ),
            None,
        )

        if step_index is None:
            continue

        if record["file_exists"]:
            tweet_file_counts[step_index] += 1

        counts[step_index] += record["n_msgs"]

    missing_steps = [
        ts[i].isoformat()
        for i, count in enumerate(counts)
        if count == 0
    ]

    return counts, tweet_file_counts, missing_steps


print("=" * 75)
print("STOCKNET SAMPLE REJECTION DIAGNOSTIC")
print("=" * 75)

stocks = list(mappings.stock_id_dict.keys())

for phase in PHASES:

    print(f"\nPHASE: {phase.upper()}")

    phase_counter = Counter()
    phase_start = time.time()

    for stock_index, stock in enumerate(stocks, start=1):

        records = price_reader.read(stock)

        start_date, end_date = builder.PHASE_DATES[phase]

        target_dates = [
            d for d in records
            if start_date <= d.isoformat() < end_date
        ]

        stock_counter = Counter()

        for target_date in target_dates:

            price_status, price_info = diagnose_price_timeline(
                records,
                target_date,
            )

            if price_status != "PRICE_OK":

                status = price_status
                counts = []
                file_counts = []
                missing_steps = []

            else:

                counts, file_counts, missing_steps = (
                    diagnose_tweet_alignment(
                        stock,
                        target_date,
                        price_info["ts"],
                    )
                )

                if missing_steps:
                    status = "TWEET_ALIGNMENT_FAILED"
                else:
                    status = "VALID_SAMPLE"

            # Cross-check with the actual existing sample builder.
            actual_sample = builder.build(
                stock=stock,
                target_date=target_date,
                price_records=records,
            )

            actual_valid = actual_sample is not None
            diagnostic_valid = status == "VALID_SAMPLE"

            if actual_valid != diagnostic_valid:
                status = "DIAGNOSTIC_MISMATCH"

            stock_counter[status] += 1
            phase_counter[status] += 1

            details.append({
                "phase": phase,
                "stock": stock,
                "target_date": target_date.isoformat(),
                "movement": records[target_date]["movement"],
                "status": status,
                "T": len(price_info["ts"]) if price_info else "",
                "trading_dates": json.dumps(
                    [d.isoformat() for d in price_info["ts"]]
                ) if price_info else "",
                "messages_per_step": json.dumps(counts),
                "tweet_files_per_step": json.dumps(file_counts),
                "missing_trading_steps": json.dumps(missing_steps),
                "actual_builder_valid": actual_valid,
            })

        summary.append({
            "phase": phase,
            "stock": stock,
            "target_dates": len(target_dates),
            "valid_samples": stock_counter["VALID_SAMPLE"],
            "price_date_missing": stock_counter["PRICE_DATE_MISSING"],
            "price_movement_filter": stock_counter["PRICE_MOVEMENT_FILTER"],
            "price_timeline_invalid": stock_counter["PRICE_TIMELINE_INVALID"],
            "tweet_alignment_failed": stock_counter["TWEET_ALIGNMENT_FAILED"],
            "diagnostic_mismatches": stock_counter["DIAGNOSTIC_MISMATCH"],
        })

        print(
            f"[{stock_index:02d}/{len(stocks)}] {stock:8s} | "
            f"Targets: {len(target_dates):5d} | "
            f"Valid: {stock_counter['VALID_SAMPLE']:5d} | "
            f"Tweet rejected: "
            f"{stock_counter['TWEET_ALIGNMENT_FAILED']:5d}"
        )

    print("\nPhase rejection summary:")

    for reason, count in sorted(phase_counter.items()):
        print(f"{reason:30s}: {count}")

    print("Time:", round(time.time() - phase_start, 2), "seconds")


# Save detailed target-date report.
details_path = ROOT / "m2_workspace" / "rejection_details.csv"

with details_path.open("w", newline="", encoding="utf-8") as file:
    writer = csv.DictWriter(
        file,
        fieldnames=list(details[0].keys()) if details else [],
    )
    writer.writeheader()
    writer.writerows(details)


# Save stock-wise summary.
summary_path = ROOT / "m2_workspace" / "rejection_summary.csv"

with summary_path.open("w", newline="", encoding="utf-8") as file:
    writer = csv.DictWriter(
        file,
        fieldnames=list(summary[0].keys()) if summary else [],
    )
    writer.writeheader()
    writer.writerows(summary)


print("\n" + "=" * 75)
print("FINAL DIAGNOSTIC SUMMARY")
print("=" * 75)

overall = Counter(row["status"] for row in details)

for reason, count in sorted(overall.items()):
    print(f"{reason:30s}: {count}")

print("\nDetailed report:", details_path)
print("Summary report:", summary_path)
print("Total target dates examined:", len(details))
print("Total runtime:", round(time.time() - overall_start, 2), "seconds")
