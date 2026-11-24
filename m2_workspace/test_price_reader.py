
from pathlib import Path

from price_reader import PriceReader


PROJECT_ROOT = Path(__file__).resolve().parents[1]

PRICE_ROOT = (
    PROJECT_ROOT
    / "stocknet-dataset"
    / "price"
    / "preprocessed"
)

reader = PriceReader(PRICE_ROOT)

prices = reader.read("AAPL")

print("Total price records:", len(prices))

# Display the five most recent trading dates.
for trading_date in sorted(prices, reverse=True)[:5]:
    record = prices[trading_date]

    print(
        trading_date,
        "Movement:", record["movement"],
        "Features:", reader.get_price_features(record),
        "Label:", reader.get_label(record),
        "Eligible:", reader.is_eligible_target(record),
    )

# Verify the known target from the M3 handoff.
target_date = __import__("datetime").date(2014, 1, 7)

assert target_date in prices

print("\n2014-01-07 verification:")
print(prices[target_date])
print("Features:", reader.get_price_features(prices[target_date]))
print("Label:", reader.get_label(prices[target_date]))
print("Eligible:", reader.is_eligible_target(prices[target_date]))

print("\nPriceReader checks passed.")
