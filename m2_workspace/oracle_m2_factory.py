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


def make_builder(stock_symbols=None):

    mappings = StockNetMappings(
        config_path=CODE / "src" / "config.yml",
        vocab_path=CODE / "res" / "vocab.txt",
    )

    price_reader = PriceReader(
        DATA / "price" / "preprocessed"
    )

    tweet_reader = TweetReader(
        tweet_root=DATA / "tweet" / "preprocessed",
        vocab_path=CODE / "res" / "vocab.txt",
    )

    return StockNetSampleBuilder(
        mappings=mappings,
        price_reader=price_reader,
        tweet_reader=tweet_reader,
        max_n_days=5,
        max_n_msgs=30,
        max_n_words=40,
        shuffle=False,
    )