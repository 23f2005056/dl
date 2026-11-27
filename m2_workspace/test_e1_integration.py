import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(0, str(ROOT / "m1_workspace"))
sys.path.insert(0, str(ROOT / "m2_workspace"))

import numpy as np
import torch

from oracle_m2_factory import make_builder
from batch_builder import StockNetBatchBuilder
from mie import MarketInformationEncoder
from e1_integration import E1Integration

builder = make_builder()

batch_builder = StockNetBatchBuilder(
    builder,
    list(builder.mappings.stock_id_dict.keys()),
    batch_size=32
)

np_batch = next(batch_builder.iter_batches("train"))

words = torch.from_numpy(np_batch["word_batch"].copy()).long()
prices = torch.from_numpy(np_batch["price_batch"].copy()).float()
n_words = torch.from_numpy(np_batch["n_words_batch"].copy()).long()
n_msgs = torch.from_numpy(np_batch["n_msgs_batch"].copy()).long()
lengths = torch.from_numpy(np_batch["T_batch"].copy()).long()

assert prices.shape == (32, 5, 3)
assert lengths.shape == (32,)
assert np.isfinite(prices.numpy()).all()

mie = MarketInformationEncoder(vocab_size=29867)

x_original, c_t = mie(
    words=words,
    prices=prices,
    n_words=n_words,
    n_msgs=n_msgs
)

assert c_t.shape == (32, 5, 100)
assert x_original.shape == (32, 5, 103)

e1 = E1Integration()
x_e1 = e1(c_t, prices, lengths)

assert x_e1.shape == (32, 5, 150)
assert torch.isfinite(x_e1).all()

for i, length in enumerate(lengths):
    assert torch.allclose(
        x_e1[i, int(length):, :],
        torch.zeros_like(x_e1[i, int(length):, :]),
        atol=1e-6
    )

loss = x_e1.square().mean()
loss.backward()

for name, parameter in e1.named_parameters():
    assert parameter.grad is not None, name
    assert torch.isfinite(parameter.grad).all(), name

print("Real M2 batch: PASSED")
print("MIE tweet representation:", tuple(c_t.shape))
print("Original E0 representation:", tuple(x_original.shape))
print("E1 combined representation:", tuple(x_e1.shape))
print("Variable sequence padding: PASSED")
print("Backward pass: PASSED")
print("GRU parameter gradients: PASSED")
print("E1 integration test: PASSED")
