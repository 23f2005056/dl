
from pathlib import Path

from stocknet_mappings import StockNetMappings


PROJECT_ROOT = Path(__file__).resolve().parents[1]

mappings = StockNetMappings(
    config_path=PROJECT_ROOT / "stocknet-code" / "src" / "config.yml",
    vocab_path=PROJECT_ROOT / "stocknet-code" / "res" / "vocab.txt",
)

# Check stock mappings.
assert len(mappings.stock_symbols) == 88
assert len(mappings.stock_id_dict) == 88
assert mappings.stock_id_dict["XOM"] == 0
assert mappings.stock_id_dict["AAPL"] == 10

# Check vocabulary mappings.
assert len(mappings.vocab) == 29866
assert len(mappings.vocab_id_dict) == 29867
assert mappings.vocab_id_dict["UNK"] == 0
assert mappings.vocab_id_dict["fawn"] == 1
assert mappings.vocab_id_dict["jawbone"] == 29866

# Check unknown-token handling.
assert mappings.convert_token_to_id(
    "not_in_vocabulary",
    mappings.vocab_id_dict,
) == 0

# Check known-token handling.
assert mappings.convert_token_to_id(
    "AAPL",
    mappings.stock_id_dict,
) == 10

print("Stock count:", len(mappings.stock_symbols))
print("Vocabulary count:", len(mappings.vocab))
print("Model vocabulary size:", len(mappings.vocab_id_dict))
print("AAPL stock ID:", mappings.stock_id_dict["AAPL"])
print("UNK ID:", mappings.vocab_id_dict["UNK"])
print("All mapping checks passed.")
