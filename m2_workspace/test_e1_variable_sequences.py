
import numpy as np
import torch

from oracle_m2_factory import make_builder
from batch_builder import StockNetBatchBuilder
from price_encoder import GRUPriceEncoder

# 1. Load a real M2 training batch
builder = make_builder()

batch_builder = StockNetBatchBuilder(
    builder,
    list(builder.mappings.stock_id_dict.keys()),
    batch_size=32
)

batch = next(batch_builder.iter_batches("train"))

# 2. Extract prices and sequence lengths
price_array = batch["price_batch"]
lengths_array = batch["T_batch"]

assert price_array.shape == (32, 5, 3)
assert lengths_array.shape == (32,)
assert np.isfinite(price_array).all()

price = torch.from_numpy(price_array.copy())
lengths = torch.from_numpy(lengths_array.copy())

# 3. Initialise E1
encoder = GRUPriceEncoder(
    input_size=3,
    hidden_size=50
)

# 4. Forward pass with actual sequence lengths
output = encoder(price, lengths)

assert output.shape == (32, 5, 50)
assert torch.isfinite(output).all()

# 5. Check padded positions
for i, length in enumerate(lengths_array):
    padded_output = output[i, int(length):, :]

    assert torch.allclose(
        padded_output,
        torch.zeros_like(padded_output)
    )

# 6. Backward pass
price.requires_grad_(True)

# Re-run forward pass with gradients enabled
output = encoder(price, lengths)
loss = output.square().mean()
loss.backward()

assert price.grad is not None
assert torch.isfinite(price.grad).all()

for name, parameter in encoder.named_parameters():
    assert parameter.grad is not None
    assert torch.isfinite(parameter.grad).all()

print("Real M2 batch loaded: PASSED")
print("Input shape:", tuple(price.shape))
print("Sequence lengths:", np.unique(lengths_array))
print("Output shape:", tuple(output.shape))
print("Padded positions check: PASSED")
print("Backward pass: PASSED")
print("GRU parameter gradients: PASSED")
print("E1 variable-length test: PASSED")
