"""Check the E2 draft on genuine M2-aligned StockNet price samples."""

import sys
from pathlib import Path

import torch

from src.models.tcn import PriceTCN


PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "m2_workspace"))

from price_reader import PriceReader  # noqa: E402
from sample_builder import StockNetSampleBuilder  # noqa: E402
from stocknet_mappings import StockNetMappings  # noqa: E402
from tweet import TweetReader  # noqa: E402


mappings = StockNetMappings(
    PROJECT / "stocknet-code/src/config.yml",
    PROJECT / "stocknet-code/res/vocab.txt",
)
samples = StockNetSampleBuilder(
    mappings,
    PriceReader(PROJECT / "stocknet-dataset/price/preprocessed"),
    TweetReader(),
    shuffle=False,
).iter_samples("AAPL", "train")
real_prices = []
dates = []
for sample in samples:
    if sample["T"] == 5:
        real_prices.append(sample["prices"])
        dates.append(sample["main_target_date"])
    if len(real_prices) == 2:
        break
assert len(real_prices) == 2, "need two genuine five-step AAPL samples"

torch.manual_seed(0)
encoder = PriceTCN()
prices = torch.tensor(real_prices, dtype=torch.float32, requires_grad=True)
lengths = torch.tensor([5, 5], dtype=torch.int64)
encoded = encoder(prices, lengths)
assert encoded.shape == (2, 5, 50)
assert torch.isfinite(encoded).all()

# A time step must have no gradient path to any later input price.
for step in range(5):
    gradient = torch.autograd.grad(
        encoded[:, step, :].sum(), prices, retain_graph=True
    )[0]
    assert torch.count_nonzero(gradient[:, step + 1 :, :]) == 0

# The two-layer receptive field spans at most four inputs.
last_gradient = torch.autograd.grad(encoded[:, 4, :].sum(), prices)[0]
assert torch.count_nonzero(last_gradient[:, 0, :]) == 0

encoder.zero_grad()
encoder(prices, lengths).sum().backward()
assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in encoder.parameters())

# Short samples must remain zero after their true length. Padding values
# cannot influence a real output, even if they are accidentally nonzero.
short_lengths = torch.tensor([3, 5], dtype=torch.int64)
short_prices = prices.detach().clone()
short_output = encoder(short_prices, short_lengths)
assert torch.count_nonzero(short_output[0, 3:]) == 0
short_prices[0, 3:] = 999.0
assert torch.equal(encoder(short_prices, short_lengths), short_output)

print("E2 TCN passed real-price shape, causal-gradient, padding, and backward checks.")
print("AAPL target dates:", dates)
