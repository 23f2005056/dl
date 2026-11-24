import torch
from torch import nn
from batch import MAX_MSGS, MAX_WORDS, PRICE_DIM, VOCAB_SIZE


class DummyModel(nn.Module):
    def __init__(self, emb=50, h=64):
        super().__init__()
        self.word_emb = nn.Embedding(VOCAB_SIZE, emb, padding_idx=0)
        self.gru = nn.GRU(emb + PRICE_DIM, h, batch_first=True)
        self.out = nn.Linear(h, 2)

    def forward(self, batch, step=0):
        w = self.word_emb(batch["word"])                                   # B,T,M,W,E
        dev = w.device
        wmask = (torch.arange(MAX_WORDS, device=dev) < batch["n_words"][..., None]).float()
        msg = (w * wmask[..., None]).sum(3) / wmask.sum(3).clamp(min=1)[..., None]
        mmask = (torch.arange(MAX_MSGS, device=dev) < batch["n_msgs"][..., None]).float()
        day = (msg * mmask[..., None]).sum(2) / mmask.sum(2).clamp(min=1)[..., None]
        out, _ = self.gru(torch.cat([day, batch["price"]], -1))
        last = out[torch.arange(len(out), device=dev), batch["T"] - 1]
        return {"logits": self.out(last)}