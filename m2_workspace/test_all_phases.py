
exec(open(
    str(__import__("pathlib").Path(__file__).parent / "test_batch_builder.py")
).read().split("# Generate one training batch.")[0])

for phase in ["train", "dev", "test"]:
    batch = next(batch_builder.iter_batches(phase))

    expected_size = 32 if phase == "train" else 1

    assert batch["batch_size"] == expected_size
    assert batch["stock_batch"].shape == (expected_size,)
    assert batch["y_batch"].shape == (expected_size, 5, 2)

    print(
        phase,
        "| batch size:", batch["batch_size"],
        "| sequence lengths:", batch["T_batch"].tolist(),
    )

print("All phase checks passed.")
