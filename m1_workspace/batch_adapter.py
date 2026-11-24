import torch
import numpy as np

def prepare_torch_batch(np_batch: dict, device: torch.device) -> dict:
    """
    Converts M2's NumPy batch dictionary into device-agnostic PyTorch tensors.
    """
    batch = {
        # Text input to MIE: [B, 5, 30, 40]
        "tweet_ids": torch.from_numpy(np_batch["word_batch"]).long().to(device),

        # Price input to MIE: [B, 5, 3]
        "price_features": torch.from_numpy(np_batch["price_batch"]).float().to(device),

        # Padding lengths for text masks
        "n_msgs": torch.from_numpy(np_batch["n_msgs_batch"]).long().to(device),      # [B, 5]
        "n_words": torch.from_numpy(np_batch["n_words_batch"]).long().to(device),    # [B, 5, 30]

        # Sequence length for variable trading days (2 to 5)
        "T_lengths": torch.from_numpy(np_batch["T_batch"]).long().to(device),        # [B]

        # Historical one-hot labels for auxiliary loss: [B, 5, 2]
        "y_historical": torch.from_numpy(np_batch["y_batch"]).float().to(device),

        # Binary target labels: [B] (0 for fall, 1 for rise)
        "targets": torch.from_numpy(
            np_batch["targets"] if "targets" in np_batch 
            else (np_batch["main_mv_percent_batch"] > 1e-7).astype(np.int64)
        ).long().to(device),

        # Metadata preserved for M3 evaluation CSV export (strings, stay on CPU)
        "stock_names": np_batch.get("stock_names", []),
        "target_dates": np_batch.get("target_dates", []),
    }

    # Stock symbol token index in each tweet for faithful BiGRU state extraction (Model.py:173-181)
    if "ss_index_batch" in np_batch:
        batch["ss_index"] = torch.from_numpy(np_batch["ss_index_batch"]).long().to(device) # [B, 5, 30]

    return batch
