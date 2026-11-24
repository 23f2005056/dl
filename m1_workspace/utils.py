import os
import random
import numpy as np
import torch
import torch.nn as nn

def seed_everything(seed: int = 42):
    """
    Sets random seed across all libraries for deterministic reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def clip_gradients(model: nn.Module, max_norm: float = 15.0) -> float:
    """
    Clips gradients by global norm.
    StockNet config.yml line 19 specifies clip: 15.0.
    """
    return torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_norm).item()

def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    dev_mcc: float,
    dev_acc: float,
    filepath: str
):
    """
    Saves a model checkpoint for validation-based selection.
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    state = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "dev_mcc": dev_mcc,
        "dev_acc": dev_acc,
    }
    torch.save(state, filepath)

def load_checkpoint(filepath: str, model: nn.Module, optimizer: torch.optim.Optimizer = None):
    """
    Loads a saved checkpoint.
    """
    checkpoint = torch.load(filepath, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return checkpoint
