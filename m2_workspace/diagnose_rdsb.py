
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent

sys.path.insert(0, str(ROOT / "m2_workspace"))
sys.path.insert(0, str(ROOT / "m3_workspace"))

from datetime import date

from price_reader import PriceReader
from tweet import TweetReader
from date_alignment import (
    build_price_timeline,
    align_tweets_for_stock,
)

DATA = ROOT / "stocknet-dataset"
CODE = ROOT / "stocknet-code"

stock = "RDS-B"

price_reader = PriceReader(
    DATA / "price" / "preprocessed"
)

tweet_reader = TweetReader(
    tweet_root=DATA / "tweet" / "preprocessed",
    vocab_path=CODE / "res" / "vocab.txt",
)

price_records = price_reader.read(stock)

tweet_dir = DATA / "tweet" / "preprocessed" / stock

tweet_dates = sorted(
    date.fromisoformat(path.name)
    for path in tweet_dir.iterdir()
    if path.is_file()
)

print("STOCK:", stock)
print("TWEET FILE COUNT:", len(tweet_dates))

for tweet_date in tweet_dates:

    print("\n" + "=" * 55)
    print("TWEET DATE:", tweet_date)

    # Original alignment assigns a tweet to the first
    # trading date strictly after its calendar date.
    future_trading_dates = [
        d for d in price_records
        if d > tweet_date
    ]

    if not future_trading_dates:
        print("No subsequent trading date.")
        continue

    candidate_target = min(future_trading_dates)

    print("Next trading date:", candidate_target)

    price_info = build_price_timeline(
        price_records=price_records,
        target_date=candidate_target,
        max_n_days=5,
    )

    if price_info is None:
        print("PRICE RESULT: Target rejected.")
        continue

    print("PRICE RESULT: Eligible")
    print("T:", price_info["T"])
    print("Trading dates:", price_info["ts"])

    tweet_info = align_tweets_for_stock(
        stock=stock,
        ts=price_info["ts"],
        target_date=candidate_target,
        tweet_reader=tweet_reader,
        max_n_days=5,
        max_n_msgs=30,
        max_n_words=40,
    )

    if tweet_info is None:
        print("TWEET ALIGNMENT: Rejected")
    else:
        print("TWEET ALIGNMENT: Passed")
        print("Messages per step:", tweet_info["n_msgs"])

print("\nRDS-B diagnostic completed.")
