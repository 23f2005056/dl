"""E2: MIE tweet vector c_t [B,5,100] + M3 causal PriceTCN [B,5,50] -> VMD input [B,5,150]."""
import torch
from torch import nn
from src.models.tcn import PriceTCN   # m3_workspace is put on sys.path by train.py


class E2Integration(nn.Module):
    def __init__(self):
        super().__init__()
        self.price_encoder = PriceTCN()

    def forward(self, c_t, price, lengths):
        return torch.cat([c_t, self.price_encoder(price, lengths)], dim=-1)