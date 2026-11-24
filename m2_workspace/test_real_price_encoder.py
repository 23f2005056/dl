
import numpy as np
import torch

from oracle_m2_factory import make_builder
from batch_builder import StockNetBatchBuilder
from price_encoder import OriginalPriceEncoder


# 1. Initialize the validated M2 data pipeline
builder = make_builder()

batch_builder = StockNetBatchBuilder(
    builder,
    list(builder.mappings.stock_id_dict.keys()),
    batch_size=32
)

# 2. Generate one real training batch
batch = next(batch_builder.iter_batches("train"))

# 3. Extract actual price data
price_array = batch["price_batch"]

assert price_array.shape == (32, 5, 3)
assert price_array.dtype == np.float32
assert np.isfinite(price_array).all()

# 4. Convert NumPy data into a PyTorch tensor
price = torch.from_numpy(price_array.copy())
price.requires_grad_(True)

# 5. Run E0
encoder = OriginalPriceEncoder()
output = encoder(price)

# 6. Verify output
assert output.shape == (32, 5, 3)
assert torch.equal(output, price)

# 7. Verify backward pass
loss = output.square().mean()
loss.backward()

assert price.grad is not None
assert torch.isfinite(price.grad).all()

print("Real M2 batch loaded: PASSED")
print("Input shape:", tuple(price.shape))
print("Output shape:", tuple(output.shape))
print("Price values preserved: PASSED")
print("Backward pass: PASSED")
print("Gradient check: PASSED")
print("Sample stock names:", batch["stock_names"][:5])
print("Sample target dates:", batch["target_dates"][:5])
print("E0 real-data test: PASSED")
