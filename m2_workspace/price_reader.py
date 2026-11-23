
"""Read StockNet's original preprocessed price files."""

from datetime import date
from pathlib import Path


class PriceReader:
    """Load and validate one stock's preprocessed price history."""

    def __init__(self, price_root):
        self.price_root = Path(price_root)

    def read(self, stock):
        """
        Return:
            {
                date: {
                    "movement": float,
                    "open": float,
                    "high": float,
                    "low": float,
                    "close": float,
                    "volume": float
                }
            }

        The source file is tab-separated and stored newest first.
        """
        path = self.price_root / f"{stock}.txt"

        if not path.is_file():
            raise FileNotFoundError(f"Price file not found: {path}")

        records = {}

        with path.open("r", encoding="utf-8") as file:
            for line_number, line in enumerate(file, start=1):
                line = line.strip()

                if not line:
                    continue

                columns = line.split("\t")

                if len(columns) != 7:
                    raise ValueError(
                        f"{path}, line {line_number}: "
                        f"expected 7 tab-separated columns, "
                        f"found {len(columns)}"
                    )

                try:
                    trading_date = date.fromisoformat(columns[0])

                    record = {
                        "movement": float(columns[1]),
                        "open": float(columns[2]),
                        "high": float(columns[3]),
                        "low": float(columns[4]),
                        "close": float(columns[5]),
                        "volume": float(columns[6]),
                    }
                except ValueError as exc:
                    raise ValueError(
                        f"Invalid data in {path}, line {line_number}: {exc}"
                    ) from exc

                if trading_date in records:
                    raise ValueError(
                        f"Duplicate date {trading_date} in {path}"
                    )

                records[trading_date] = record

        if not records:
            raise ValueError(f"No price records found in {path}")

        return records

    @staticmethod
    def get_price_features(record):
        """Return the original model's three price features: high, low, close."""
        return [
            record["high"],
            record["low"],
            record["close"],
        ]

    @staticmethod
    def get_label(record):
        """Original binary label: 0 = fall, 1 = rise."""
        return int(record["movement"] > 1e-7)

    @staticmethod
    def is_eligible_target(record):
        """Apply the original StockNet neutral-movement exclusion."""
        movement = record["movement"]
        return not (-0.005 <= movement < 0.0055)
