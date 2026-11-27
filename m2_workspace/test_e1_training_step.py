
import numpy as np
import torch

from oracle_m2_factory import make_builder
from batch_builder import StockNetBatchBuilder
from price_encoder import GRUPriceEncoder

# Load a real M2 batch
builder = make_builder()

batch_builder = StockNetBatchBuilder(
    builder,
    list(builder.mappings.stock_id_dict.keys()),
    batch_size=32
)

batch = next(batch_builder.iter_batches("train"))

price = torch.from_numpy(batch["price_batch"].copy())
lengths = torch.from_numpy(batch["T_batch"].copy())

# Initialise E1
encoder = GRUPriceEncoder(input_size=3, hidden_size=50)
optimizer = torch.optim.Adam(encoder.parameters(), lr=0.001)

# Save initial parameters
initial_parameters = {
    name: parameter.detach().clone()
    for name, parameter in encoder.named_parameters()
}

# Forward pass
output = encoder(price, lengths)

# Test loss only; not the StockNet prediction objective
loss = output.square().mean()

# Backward and optimizer update
optimizer.zero_grad()
loss.backward()
optimizer.step()

# Verify parameter updates
changed = []

for name, parameter in encoder.named_parameters():
    assert parameter.grad is not None
    assert torch.isfinite(parameter.grad).all()

    if not torch.equal(parameter.detach(), initial_parameters[name]):
        changed.append(name)

assert len(changed) == len(initial_parameters)

print("Real M2 batch: PASSED")
print("Sequence lengths:", np.unique(lengths.numpy()))
print("Output shape:", tuple(output.shape))
print("Loss:", loss.item())
print("Backward pass: PASSED")
print("Optimizer update: PASSED")
print("Changed parameter tensors:", len(changed))
print("E1 variable-length training test: PASSED")
