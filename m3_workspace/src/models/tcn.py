"""E2 causal price encoder for StockNet.

Input:  normalized [high, low, close] prices, oldest step first, [B, 5, 3].
Output: one 50-dimensional price representation per step, [B, 5, 50].

Each convolution is padded only on the left. With kernel size 2 and
dilations 1 and 2, an output can depend on at most its current input and
the preceding three steps. This module is not connected to E0.
"""

import torch
from torch import nn
from torch.nn import functional as F


class PriceTCN(nn.Module):
    """Two-layer causal dilated Conv1D price encoder for the E2 ablation."""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(3, 32, kernel_size=2, dilation=1)
        self.conv2 = nn.Conv1d(32, 50, kernel_size=2, dilation=2)
        self.activation = nn.ReLU()

    def forward(self, prices: torch.Tensor, T_lengths: torch.Tensor) -> torch.Tensor:
        if prices.ndim != 3 or prices.shape[1:] != (5, 3):
            raise ValueError("prices must have shape [B, 5, 3]")
        if not torch.is_floating_point(prices):
            raise TypeError("prices must be a floating-point tensor")
        if not torch.isfinite(prices).all():
            raise ValueError("prices must be finite")
        if T_lengths.ndim != 1 or T_lengths.shape[0] != prices.shape[0]:
            raise ValueError("T_lengths must have shape [B]")
        if T_lengths.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64):
            raise TypeError("T_lengths must contain integers")
        if torch.any(T_lengths < 1) or torch.any(T_lengths > 5):
            raise ValueError("T_lengths must be between 1 and 5")

        valid = torch.arange(5, device=prices.device)[None, :] < T_lengths.to(prices.device)[:, None]
        prices = prices * valid.unsqueeze(-1)
        x = prices.transpose(1, 2)  # [B, 3, 5]
        x = self.activation(self.conv1(F.pad(x, (1, 0))))
        x = self.activation(self.conv2(F.pad(x, (2, 0))))
        return x.transpose(1, 2) * valid.unsqueeze(-1)  # [B, 5, 50]


# Preserve the draft's import name for existing local checks.
CausalPriceTCN = PriceTCN
