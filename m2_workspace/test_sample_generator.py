
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent

sys.path.insert(0, str(ROOT / "m2_workspace"))
sys.path.insert(0, str(ROOT / "m3_workspace"))

from stocknet_mappings import StockNetMappings
from price_reader import PriceReader
from tweet import TweetReader
from sample_builder import StockNetSampleBuilder


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
    shuffle=False,  # Easier to inspect repeatable date order.
)

# Generate AAPL training samples.
samples = builder.iter_samples("AAPL", "train")

first_sample = next(samples, None)

assert first_sample is not None, "No eligible AAPL training sample found."

assert first_sample["stock"] == mappings.stock_id_dict["AAPL"]

assert (
    "2014-01-01"
    <= first_sample["main_target_date"]
    < "2015-08-01"
)

print("First sample date:", first_sample["main_target_date"])
print("Stock ID:", first_sample["stock"])
print("T:", first_sample["T"])
print("Message counts:", first_sample["n_msgs"])
print("Sample generator test passed.")
