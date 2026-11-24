
import torch

from price_encoder import OriginalPriceEncoder


# 1. Create E0
encoder = OriginalPriceEncoder()

# 2. Simulate a batch of 32 samples
price = torch.randn(
    32, 5, 3,
    dtype=torch.float32,
    requires_grad=True
)

# 3. Forward pass
output = encoder(price)

# 4. Verify shape
assert output.shape == (32, 5, 3)

# 5. Verify original values are preserved
assert torch.equal(output, price)

# 6. Backward pass
loss = output.square().mean()
loss.backward()

# 7. Verify gradients
assert price.grad is not None
assert torch.isfinite(price.grad).all()

print("E0 forward pass: PASSED")
print("E0 shape:", tuple(output.shape))
print("E0 value preservation: PASSED")
print("E0 backward pass: PASSED")
print("E0 gradient check: PASSED")
