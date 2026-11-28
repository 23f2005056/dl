"""Variational Movement Decoder: port of Model.py _create_vmd_with_zh_rec (386-503) and _z (750-759)."""
import torch
from torch import nn


class VMD(nn.Module):
    def __init__(self, x_size=103, h_size=150, z_size=150, g_size=50, y_size=2, dropout=0.3):
        super().__init__()
        self.z_size = z_size
        self.drop = nn.Dropout(dropout)                       # dropout_vmd_in
        self.gru = nn.GRU(x_size, h_size, batch_first=True)   # vmd_cell_type: gru
        self.h_z_prior = nn.Linear(x_size + h_size + z_size, z_size)
        self.prior_mu, self.prior_lv = nn.Linear(z_size, z_size), nn.Linear(z_size, z_size)
        self.h_z_post = nn.Linear(x_size + h_size + y_size + z_size, z_size)
        self.post_mu, self.post_lv = nn.Linear(z_size, z_size), nn.Linear(z_size, z_size)
        self.g = nn.Linear(h_size + z_size, g_size)

    def forward(self, x, T, y):
        """x [B,D,x_size], T [B], y [B,D,2] one-hot -> g [B,D,50], kl [B,D]."""
        B, D, _ = x.shape
        x = self.drop(x)
        packed = nn.utils.rnn.pack_padded_sequence(x, T.cpu(), batch_first=True, enforce_sorted=False)
        h, _ = self.gru(packed)
        h, _ = nn.utils.rnn.pad_packed_sequence(h, batch_first=True, total_length=D)

        # Original starts from random z; in eval we use zeros + posterior means so outputs are deterministic.
        z_prev = torch.randn(B, self.z_size, device=x.device) if self.training else x.new_zeros(B, self.z_size)
        z_prior, z_post, kl = [], [], []
        for t in range(D):
            hp = torch.tanh(self.h_z_prior(torch.cat([x[:, t], h[:, t], z_prev], -1)))
            mu_p, lv_p = self.prior_mu(hp), self.prior_lv(hp).clamp(-10, 10)
            hq = torch.tanh(self.h_z_post(torch.cat([x[:, t], h[:, t], y[:, t], z_prev], -1)))
            mu_q, lv_q = self.post_mu(hq), self.post_lv(hq).clamp(-10, 10)
            zq = mu_q + torch.exp(0.5 * lv_q) * torch.randn_like(mu_q) if self.training else mu_q
            kl.append(0.5 * (lv_p - lv_q + (lv_q.exp() + (mu_q - mu_p) ** 2) / lv_p.exp() - 1).sum(-1))
            z_prior.append(mu_p)          # prior z = its mean (Model.py:756)
            z_post.append(zq)
            z_prev = zq
        z_prior, z_post, kl = (torch.stack(v, 1) for v in (z_prior, z_post, kl))

        g = torch.tanh(self.g(torch.cat([h, z_post], -1)))
        if not self.training:
            # Target day uses the prior, never the true label (Model.py:485-498).
            i, t = torch.arange(B, device=x.device), T - 1
            g = g.clone()
            g[i, t] = torch.tanh(self.g(torch.cat([h[i, t], z_prior[i, t]], -1)))
        return g, kl