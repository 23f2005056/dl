import torch
import torch.nn as nn

from price_encoder import GRUPriceEncoder


class E1Integration(nn.Module):
    """
    E1 integration:
    MIE tweet representation [B,T,100]
    +
    GRU price representation [B,T,50]
    =
    VMD input [B,T,150]
    """

    def __init__(self, tweet_dim=100, price_dim=3, hidden_size=50):
        super().__init__()

        self.price_encoder = GRUPriceEncoder(
            input_size=price_dim,
            hidden_size=hidden_size
        )

        self.output_dim = tweet_dim + hidden_size

    def forward(self, c_t, price, lengths):

        if c_t.ndim != 3:
            raise ValueError("Expected c_t with shape [B,T,100].")

        if c_t.shape[-1] != 100:
            raise ValueError("Expected 100-dimensional tweet features.")

        if c_t.shape[:2] != price.shape[:2]:
            raise ValueError("Tweet and price batch dimensions must match.")

        price_repr = self.price_encoder(price, lengths)

        x_t = torch.cat([c_t, price_repr], dim=-1)

        if x_t.shape[-1] != self.output_dim:
            raise ValueError("Incorrect E1 output dimension.")

        return x_t
