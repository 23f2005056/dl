
"""Load StockNet stock-symbol and vocabulary mappings."""

import json
from pathlib import Path

import yaml


class StockNetMappings:
    def __init__(self, config_path, vocab_path):
        self.config_path = Path(config_path)
        self.vocab_path = Path(vocab_path)

        # Load the original StockNet configuration.
        with self.config_path.open("r", encoding="utf-8") as file:
            config = yaml.safe_load(file)

        # Flatten stock groups while preserving their YAML order.
        stock_groups = config["stocks"].values()
        self.stock_symbols = [
            stock
            for group in stock_groups
            for stock in group
        ]

        if len(self.stock_symbols) != len(set(self.stock_symbols)):
            raise ValueError("Duplicate stock symbols found.")

        # Stock IDs start at zero.
        self.stock_id_dict = {
            stock: index
            for index, stock in enumerate(self.stock_symbols)
        }

        # Load the original vocabulary.
        self.vocab = json.loads(
            self.vocab_path.read_text(encoding="utf-8")
        )

        if not isinstance(self.vocab, list):
            raise ValueError("Vocabulary must be a JSON list.")

        if len(self.vocab) != len(set(self.vocab)):
            raise ValueError("Duplicate vocabulary tokens found.")

        if "UNK" in self.vocab:
            raise ValueError(
                "UNK already exists in the original vocabulary."
            )

        # Add UNK at index zero, preserving original word IDs.
        self.vocab_id_dict = {
            "UNK": 0,
            **{
                word: index
                for index, word in enumerate(self.vocab, start=1)
            },
        }

    @staticmethod
    def convert_token_to_id(token, mapping):
        """Return the token ID, or zero for an unknown token."""
        return mapping.get(token, 0)
