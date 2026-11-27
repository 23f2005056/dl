
import torch
import torch.nn as nn


class OriginalPriceEncoder(nn.Module):
    """
    E0: Original StockNet price pathway.

    The original StockNet implementation passes
    High, Low and Close directly into its MIE.

    This module preserves that behaviour.
    """

    def __init__(self, input_size=3):
        super().__init__()

        if input_size != 3:
            raise ValueError(
                "Original StockNet expects 3 price features."
            )

        self.input_size = input_size

    def forward(self, price):
        """
        Args:
            price: Tensor of shape [B, T, 3]

        Returns:
            Original price tensor of shape [B, T, 3]
        """

        if price.ndim != 3:
            raise ValueError(
                f"Expected 3D input [B,T,3], got {price.shape}"
            )

        if price.shape[-1] != self.input_size:
            raise ValueError(
                f"Expected 3 price features, got {price.shape[-1]}"
            )

        if not torch.is_floating_point(price):
            raise TypeError("Price input must be floating point.")

        if not torch.isfinite(price).all():
            raise ValueError("Price input contains NaN or infinity.")

        return price



class GRUPriceEncoder(nn.Module):

    def __init__(self, input_size=3, hidden_size=50):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size

        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True
        )

    def forward(self, price, lengths):

        if price.ndim != 3:
            raise ValueError(
                f"Expected [B,T,3], got {price.shape}"
            )

        if price.shape[-1] != self.input_size:
            raise ValueError("Incorrect number of price features")

        if not torch.is_floating_point(price):
            raise TypeError("Price input must be floating point.")

        if not torch.isfinite(price).all():
            raise ValueError("Price input contains NaN or infinity.")

        if lengths.ndim != 1 or lengths.shape[0] != price.shape[0]:
            raise ValueError("Invalid sequence lengths.")

        lengths = lengths.to(dtype=torch.int64, device="cpu")

        if torch.any(lengths <= 0) or torch.any(lengths > price.shape[1]):
            raise ValueError("Sequence lengths must be between 1 and T.")

        packed = nn.utils.rnn.pack_padded_sequence(
            price,
            lengths,
            batch_first=True,
            enforce_sorted=False
        )

        packed_output, hidden = self.gru(packed)

        output, _ = nn.utils.rnn.pad_packed_sequence(
            packed_output,
            batch_first=True,
            total_length=price.shape[1]
        )

        return output

