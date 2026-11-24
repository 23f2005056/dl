
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent

sys.path.insert(0, str(ROOT / "m2_workspace"))
sys.path.insert(0, str(ROOT / "m3_workspace"))

from stocknet_mappings import StockNetMappings
from price_reader import PriceReader
from tweet import TweetReader
from sample_builder import StockNetSampleBuilder
from batch_builder import StockNetBatchBuilder


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

sample_builder = StockNetSampleBuilder(
    mappings=mappings,
    price_reader=price_reader,
    tweet_reader=tweet_reader,
    max_n_days=5,
    max_n_msgs=30,
    max_n_words=40,
    shuffle=True,
)

batch_builder = StockNetBatchBuilder(
    sample_builder=sample_builder,
    stock_symbols=list(mappings.stock_id_dict.keys()),
    batch_size=32,
)

# Generate one training batch.
batch = next(batch_builder.iter_batches("train"))

assert batch["batch_size"] == 32

assert batch["stock_batch"].shape == (32,)
assert batch["T_batch"].shape == (32,)
assert batch["y_batch"].shape == (32, 5, 2)
assert batch["main_mv_percent_batch"].shape == (32,)
assert batch["mv_percent_batch"].shape == (32, 5)
assert batch["price_batch"].shape == (32, 5, 3)
assert batch["word_batch"].shape == (32, 5, 30, 40)
assert batch["ss_index_batch"].shape == (32, 5, 30)
assert batch["n_msgs_batch"].shape == (32, 5)
assert batch["n_words_batch"].shape == (32, 5, 30)

# Check data types.
assert batch["stock_batch"].dtype.name == "int32"
assert batch["y_batch"].dtype.name == "float32"
assert batch["word_batch"].dtype.name == "int32"

# Check valid sequence lengths.
assert all(1 <= T <= 5 for T in batch["T_batch"])

print("Batch size:", batch["batch_size"])
print("Stock IDs:", batch["stock_batch"])
print("Sequence lengths:", batch["T_batch"])
print("Price shape:", batch["price_batch"].shape)
print("Tweet shape:", batch["word_batch"].shape)
print("Label shape:", batch["y_batch"].shape)
print("All batch-builder checks passed.")
