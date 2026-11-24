
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

print("\n========== ACTUAL BATCH KEYS ==========")

for k, v in batch.items():
    print(
        f"{k:25} "
        f"{getattr(v, 'shape', v)!s:22} "
        f"{getattr(v, 'dtype', '')}"
    )

expected = {
    "batch_size",
    "stock_batch",
    "T_batch",
    "n_words_batch",
    "n_msgs_batch",
    "y_batch",
    "price_batch",
    "mv_percent_batch",
    "word_batch",
    "ss_index_batch",
}

print("\n========== KEY COMPARISON ==========")
print("Missing:", expected - set(batch))
print("Extra:", set(batch) - expected)

print("\n========== SEQUENCE LENGTHS ==========")
print("T_batch:", batch["T_batch"])

print("\n========== PADDING CHECK ==========")

for i, T in enumerate(batch["T_batch"]):
    print(
        f"Sample {i:2}: T={T}, "
        f"padded price rows={batch['price_batch'][i, T:].shape[0]}, "
        f"padded message counts={batch['n_msgs_batch'][i, T:]}"
    )
