
import numpy as np
import torch

from oracle_m2_factory import make_builder
from batch_builder import StockNetBatchBuilder
from price_encoder import GRUPriceEncoder

# Load a real M2 training batch
builder = make_builder()

batch_builder = StockNetBatchBuilder(
    builder,
    list(builder.mappings.stock_id_dict.keys()),
    batch_size=32
)

batch = next(batch_builder.iter_batches("train"))

# Extract price data and sequence lengths
price_array = batch["price_batch"]
lengths_array = batch["T_batch"]

assert price_array.shape == (32, 5, 3)
assert lengths_array.shape == (32,)
assert price_array.dtype == np.float32
assert np.isfinite(price_array).all()

price = torch.from_numpy(price_array.copy())
lengths = torch.from_numpy(lengths_array.copy())

price.requires_grad_(True)

# Initialise E1
encoder = GRUPriceEncoder(input_size=3, hidden_size=50)

# Forward pass
output = encoder(price, lengths)

assert output.shape == (32, 5, 50)
assert torch.isfinite(output).all()

# Backward pass
loss = output.square().mean()
loss.backward()

assert price.grad is not None
assert torch.isfinite(price.grad).all()

# Check GRU parameter gradients
for name, parameter in encoder.named_parameters():
    assert parameter.grad is not None, f"Missing gradient: {name}"
    assert torch.isfinite(parameter.grad).all(), f"Invalid gradient: {name}"

print("Real M2 batch loaded: PASSED")
print("Input shape:", tuple(price.shape))
print("Sequence lengths:", np.unique(lengths_array))
print("Output shape:", tuple(output.shape))
print("Finite output check: PASSED")
print("Backward pass: PASSED")
print("Input gradient check: PASSED")
print("GRU parameter gradient check: PASSED")
print("GRU parameter count:", sum(p.numel() for p in encoder.parameters()))
print("E1 GRU real-data test: PASSED")
