"""Batch format shared by M1 (model), M2 (data) and M4 (training).
Names follow the original Executor.py feed_dict."""
import torch

MAX_DAYS, MAX_MSGS, MAX_WORDS, PRICE_DIM = 5, 30, 40, 3
VOCAB_SIZE = 29867   # from the M3 audit
N_STOCKS = 88        # every stock in config.yml, GMRE included

# One sample is a dict with these keys (shapes without the batch dimension):
#   stock      long  []                         index into the 88 config stocks
#   T          long  []                         real number of steps (<= MAX_DAYS)
#   word       long  [MAX_DAYS,MAX_MSGS,MAX_WORDS]  0 = UNK *and* PAD -> mask with n_words
#   n_words    long  [MAX_DAYS,MAX_MSGS]
#   n_msgs     long  [MAX_DAYS]
#   ss_index   long  [MAX_DAYS,MAX_MSGS]
#   price      float [MAX_DAYS,PRICE_DIM]
#   mv_percent float [MAX_DAYS]
#   y          float [MAX_DAYS,2]               one-hot, step T-1 is the target
#   stock_symbol str, target_date str (ISO)     kept for evaluation
TENSOR_KEYS = ["stock", "T", "word", "n_words", "n_msgs", "ss_index",
               "price", "mv_percent", "y"]
META_KEYS = ["stock_symbol", "target_date"]


def collate(samples):
    batch = {k: torch.stack([torch.as_tensor(s[k]) for s in samples]) for k in TENSOR_KEYS}
    for k in META_KEYS:
        batch[k] = [s[k] for s in samples]
    return batch


def to_device(batch, device):
    return {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}


def target_label(batch):
    """Class of the target (last real) step: 0 fall, 1 rise."""
    idx = batch["T"] - 1
    return batch["y"][torch.arange(len(idx)), idx].argmax(-1)