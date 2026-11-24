
import numpy as np

from oracle_m2_factory import make_builder
from batch_builder import StockNetBatchBuilder

builder = make_builder()

batch_builder = StockNetBatchBuilder(
    builder,
    list(builder.mappings.stock_id_dict.keys()),
    batch_size=32
)

batch = next(batch_builder.iter_batches("train"))

T = batch["T_batch"]
prices = batch["price_batch"]

print("Sequence length shape:", T.shape)
print("Sequence lengths:", T)
print("Unique sequence lengths:", np.unique(T))

print("\nPrice batch shape:", prices.shape)

print("\nValid trading days per sample:")
for i in range(min(10, len(T))):
    print(
        batch["stock_names"][i],
        batch["target_dates"][i],
        "T =", T[i]
    )

print("\nSequence length inspection: PASSED")
