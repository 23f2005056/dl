"""StockNet generative ATA objective (Model.py:633-649) and KL annealing (703-712)."""
import torch
import torch.nn.functional as F


def kl_lambda(step, rate=0.005, start=0):
    return 0.0 if step < start else min(rate * step, 1.0)


def stocknet_loss(logits, aux_logits, attn, kl, y_hist, targets, T, lam, alpha=0.5):
    """loss = -mean[ obj_T + sum_t alpha * v*_t * obj_t ],  obj = log p(y) - lam * KL."""
    i = torch.arange(len(targets), device=logits.device)
    obj_T = -F.cross_entropy(logits, targets, reduction="none") - lam * kl[i, T - 1]
    ll_aux = -F.cross_entropy(aux_logits.flatten(0, 1), y_hist.argmax(-1).flatten(),
                              reduction="none").view_as(kl)
    obj_aux = ll_aux - lam * kl        # attn is already 0 for the target day and padded days
    return -(obj_T + (alpha * attn * obj_aux).sum(1)).mean()