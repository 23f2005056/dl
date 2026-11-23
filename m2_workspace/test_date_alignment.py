
import sys
from datetime import date
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# M3's tweet.py is inside m3_workspace.
sys.path.insert(0, str(PROJECT_ROOT / "m3_workspace"))

from price_reader import PriceReader
from date_alignment import (
    build_price_timeline,
    align_tweets_for_stock,
)
from tweet import TweetReader


PRICE_ROOT = (
    PROJECT_ROOT
    / "stocknet-dataset"
    / "price"
    / "preprocessed"
)

stock = "AAPL"
target_date = date(2014, 1, 7)

# 1. Read price data.
price_reader = PriceReader(PRICE_ROOT)
price_records = price_reader.read(stock)

# 2. Construct the price timeline.
timeline = build_price_timeline(
    price_records,
    target_date,
    max_n_days=5,
)

assert timeline is not None

print("PRICE TIMELINE")
print("T:", timeline["T"])
print("Trading dates:", timeline["ts"])
print("Price dates:", timeline["price_dates"])
print("Price features:", timeline["prices"])
print("Labels:", timeline["ys"])
print("Target label:", timeline["target_label"])

# 3. Verify the known M3 example.
assert timeline["ts"] == [
    date(2014, 1, 3),
    date(2014, 1, 6),
    date(2014, 1, 7),
]

assert timeline["price_dates"] == [
    date(2014, 1, 2),
    date(2014, 1, 3),
    date(2014, 1, 6),
]

# 4. Align tweets with trading dates.
tweet_reader = TweetReader()

aligned = align_tweets_for_stock(
    stock=stock,
    ts=timeline["ts"],
    target_date=target_date,
    tweet_reader=tweet_reader,
)

assert aligned is not None

print("\nTWEET ALIGNMENT")
print("Message counts:", aligned["n_msgs"])

print(
    "Word tensor shape:",
    (
        len(aligned["msgs"]),
        len(aligned["msgs"][0]),
        len(aligned["msgs"][0][0]),
    ),
)

print(
    "Stock-index shape:",
    (
        len(aligned["ss_indices"]),
        len(aligned["ss_indices"][0]),
    ),
)

print(
    "Word-count shape:",
    (
        len(aligned["n_words"]),
        len(aligned["n_words"][0]),
    ),
)

# 5. Verify tensor dimensions.
assert len(aligned["msgs"]) == 3
assert len(aligned["msgs"][0]) == 30
assert len(aligned["msgs"][0][0]) == 40

assert len(aligned["ss_indices"]) == 3
assert len(aligned["ss_indices"][0]) == 30

assert len(aligned["n_words"]) == 3
assert len(aligned["n_words"][0]) == 30

assert len(aligned["n_msgs"]) == 3
assert all(1 <= count <= 30 for count in aligned["n_msgs"])

print("\nAll date-alignment checks passed.")
