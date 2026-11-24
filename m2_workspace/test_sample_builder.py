
from datetime import date
from pathlib import Path
import sys

# Project directories.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
M2_DIR = PROJECT_ROOT / "m2_workspace"
M3_DIR = PROJECT_ROOT / "m3_workspace"

sys.path.insert(0, str(M2_DIR))
sys.path.insert(0, str(M3_DIR))

from stocknet_mappings import StockNetMappings
from price_reader import PriceReader
from tweet import TweetReader
from sample_builder import StockNetSampleBuilder


# Initialise the three data components.
mappings = StockNetMappings(
    config_path=PROJECT_ROOT / "stocknet-code" / "src" / "config.yml",
    vocab_path=PROJECT_ROOT / "stocknet-code" / "res" / "vocab.txt",
)

price_reader = PriceReader(
    PROJECT_ROOT / "stocknet-dataset" / "price" / "preprocessed"
)

tweet_reader = TweetReader(
    tweet_root=PROJECT_ROOT / "stocknet-dataset" / "tweet" / "preprocessed",
    vocab_path=PROJECT_ROOT / "stocknet-code" / "res" / "vocab.txt",
)

# Create the sample builder.
builder = StockNetSampleBuilder(
    mappings=mappings,
    price_reader=price_reader,
    tweet_reader=tweet_reader,
)

# Build the known AAPL sample.
sample = builder.build(
    stock="AAPL",
    target_date=date(2014, 1, 7),
)

assert sample is not None

# Check sample fields.
expected_keys = {
    "stock",
    "main_target_date",
    "T",
    "ys",
    "main_mv_percent",
    "mv_percents",
    "prices",
    "msgs",
    "ss_indices",
    "n_words",
    "n_msgs",
}

assert set(sample.keys()) == expected_keys

# Check known values.
assert sample["stock"] == 10
assert sample["main_target_date"] == "2014-01-07"
assert sample["T"] == 3
assert sample["main_mv_percent"] == -0.007151

# Check labels and historical movement classes.
assert sample["ys"] == [
    [1.0, 0.0],
    [0.0, 1.0],
    [1.0, 0.0],
]

assert sample["mv_percents"] == [0, 0, 1]

# Check price dimensions.
assert len(sample["prices"]) == 3
assert all(len(row) == 3 for row in sample["prices"])

# Check tweet dimensions using nested list lengths.
assert len(sample["msgs"]) == 3
assert all(len(step) == 30 for step in sample["msgs"])
assert all(
    len(message) == 40
    for step in sample["msgs"]
    for message in step
)

assert len(sample["ss_indices"]) == 3
assert all(len(step) == 30 for step in sample["ss_indices"])

assert len(sample["n_words"]) == 3
assert all(len(step) == 30 for step in sample["n_words"])

assert len(sample["n_msgs"]) == 3
assert sample["n_msgs"] == [30, 30, 23]

# Print results.
print("Stock:", sample["stock"])
print("Target date:", sample["main_target_date"])
print("T:", sample["T"])
print("Labels:", sample["ys"])
print("Main movement:", sample["main_mv_percent"])
print("Historical movement classes:", sample["mv_percents"])
print("Price shape:", (len(sample["prices"]), len(sample["prices"][0])))
print("Message counts:", sample["n_msgs"])
print("Message dimensions: (3, 30, 40)")
print("Stock-index dimensions: (3, 30)")
print("Word-count dimensions: (3, 30)")

print("\nAll sample-builder checks passed.")