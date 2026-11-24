import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(0, str(ROOT / "m1_workspace"))
sys.path.insert(0, str(ROOT / "m2_workspace"))
sys.path.insert(0, str(ROOT / "m4_workspace"))

import torch

from oracle_m2_factory import make_builder
from batch_builder import StockNetBatchBuilder
from batch_adapter import prepare_torch_batch

from stocknet import StockNet
from vmd import VMD
from losses import stocknet_loss, kl_lambda

from e1_integration import E1Integration


# 1. Load a real M2 batch
builder = make_builder()

batch_builder = StockNetBatchBuilder(
    builder,
    list(builder.mappings.stock_id_dict.keys()),
    batch_size=32
)

np_batch = next(batch_builder.iter_batches("train"))

device = torch.device("cpu")
b = prepare_torch_batch(np_batch, device)

# 2. Build the model with E1-compatible VMD
model = StockNet(
    vmd_module=VMD(x_size=150)
).to(device)

e1 = E1Integration().to(device)

model.train()
e1.train()

# 3. MIE: extract tweet representation
x_original, c_t = model.mie(
    words=b["tweet_ids"],
    prices=b["price_features"],
    n_words=b["n_words"],
    n_msgs=b["n_msgs"]
)

assert x_original.shape == (32, 5, 103)
assert c_t.shape == (32, 5, 100)

# 4. E1: encode prices and concatenate
x_t = e1(
    c_t,
    b["price_features"],
    b["T_lengths"]
)

assert x_t.shape == (32, 5, 150)

# 5. VMD: process E1 representation
g, kl = model.vmd(
    x_t,
    b["T_lengths"],
    b["y_historical"]
)

assert g.shape == (32, 5, 50)
assert kl.shape == (32, 5)

# 6. ATA: generate predictions
logits, attn, aux = model.ata(
    g,
    b["T_lengths"]
)

assert logits.shape == (32, 2)
assert attn.shape == (32, 5)
assert aux.shape == (32, 5, 2)

# 7. Calculate actual StockNet loss
lam = kl_lambda(step=200)

loss = stocknet_loss(
    logits=logits,
    aux_logits=aux,
    attn=attn,
    kl=kl,
    y_hist=b["y_historical"],
    targets=b["targets"],
    T=b["T_lengths"],
    lam=lam
)

assert torch.isfinite(loss)

# 8. Backward pass
loss.backward()

# Check E1 GRU gradients
for name, parameter in e1.named_parameters():
    assert parameter.grad is not None, name
    assert torch.isfinite(parameter.grad).all(), name

# Check VMD gradients
for name, parameter in model.vmd.named_parameters():
    assert parameter.grad is not None, name
    assert torch.isfinite(parameter.grad).all(), name

# Check MIE and ATA gradients
for module_name, module in [
    ("MIE", model.mie),
    ("ATA", model.ata)
]:
    for name, parameter in module.named_parameters():
        assert parameter.grad is not None, f"{module_name}.{name}"
        assert torch.isfinite(parameter.grad).all(), f"{module_name}.{name}"

print("Real M2 batch: PASSED")
print("MIE tweet representation:", tuple(c_t.shape))
print("E1 representation:", tuple(x_t.shape))
print("VMD representation:", tuple(g.shape))
print("KL shape:", tuple(kl.shape))
print("ATA logits:", tuple(logits.shape))
print("Auxiliary logits:", tuple(aux.shape))
print("StockNet loss:", loss.item())
print("Loss finite check: PASSED")
print("E1 gradient check: PASSED")
print("VMD gradient check: PASSED")
print("MIE gradient check: PASSED")
print("ATA gradient check: PASSED")
print("Complete E1 forward/backward test: PASSED")
