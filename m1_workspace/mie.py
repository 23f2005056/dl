import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from typing import Optional, Union, Dict

def build_glove_matrix(
    vocab_dict_or_path: Union[Dict[str, int], str, Path],
    glove_txt_path: Union[str, Path],
    embed_dim: int = 50,
    seed: int = 42
) -> torch.Tensor:
    """
    Builds a [vocab_size, embed_dim] GloVe embedding matrix matching original DataPipe.py:533.
    - Out-of-vocabulary / unseen tokens initialized with uniform random in [-1.0, 1.0].
    - PAD (index 0) set to zeros.
    - Matches exact vectors from glove.twitter.27B.50d.txt.
    """
    if isinstance(vocab_dict_or_path, (str, Path)):
        vocab_path = Path(vocab_dict_or_path)
        vocab_dict = {}
        with open(vocab_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    vocab_dict[parts[0]] = int(parts[1])
                elif len(parts) == 1:
                    vocab_dict[parts[0]] = len(vocab_dict)
    else:
        vocab_dict = vocab_dict_or_path

    vocab_size = len(vocab_dict)
    rng = np.random.RandomState(seed)
    matrix = rng.uniform(-1.0, 1.0, size=(vocab_size, embed_dim)).astype(np.float32)

    # PAD is zero
    matrix[0] = 0.0

    glove_path = Path(glove_txt_path)
    if glove_path.is_file():
        found = 0
        with open(glove_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split(" ")
                word = parts[0]
                if word in vocab_dict:
                    vec = np.array([float(v) for v in parts[1:]], dtype=np.float32)
                    if len(vec) == embed_dim:
                        matrix[vocab_dict[word]] = vec
                        found += 1
        print(f"[GloVe] Initialized {found}/{vocab_size} tokens from {glove_path.name}")
    else:
        print(f"[GloVe Warning] File {glove_path} not found. Kept random initialization.")

    return torch.from_numpy(matrix)


class MarketInformationEncoder(nn.Module):
    """
    Market Information Encoder (MIE) for StockNet.
    Corresponds to original Model.py:
      - Frozen GloVe Word Embedding: [vocab_size=29867, 50d] (trainable=False per Model.py:137)
      - Message Embedding Layer (BiGRU): hidden_size=100 in each direction
      - Stock-token extraction: (out_f[ss_index] + out_b[ss_index]) / 2 -> 100d per tweet (Model.py:173-181)
      - Corpus Embedding Layer: Attention Pooling across tweets -> c_t [100d] (Model.py:228-245)
      - Price Concatenation: [c_t (100d); p_t (3d)] -> x_t [103d]

    Input Shapes:
      words:    [B, 5, 30, 40] (int64 token IDs)
      prices:   [B, 5, 3]      (float32 normalized [H, L, C])
      n_words:  [B, 5, 30]    (int64 actual words per message)
      n_msgs:   [B, 5]        (int64 actual tweets per day)
      ss_index: [B, 5, 30]    (int64 index of stock symbol token in tweet)

    Output Shapes:
      x_t: [B, 5, 103] -> Feeds into M4's VMD recurrent cell
      c_t: [B, 5, 100] -> Tweet-only daily representation
    """
    def __init__(
        self,
        vocab_size: int = 29867,
        word_embed_size: int = 50,
        mel_h_size: int = 100,
        price_dim: int = 3,
        dropout: float = 0.3,
        pretrained_embeddings: Optional[torch.Tensor] = None,
        freeze_embeddings: bool = True
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.word_embed_size = word_embed_size
        self.mel_h_size = mel_h_size
        self.price_dim = price_dim

        # 1. Word Embedding: frozen GloVe 50d (Model.py:137 sets trainable=False)
        if pretrained_embeddings is not None:
            self.embedding = nn.Embedding.from_pretrained(
                pretrained_embeddings,
                freeze=freeze_embeddings,
                padding_idx=0
            )
        else:
            self.embedding = nn.Embedding(vocab_size, word_embed_size, padding_idx=0)
            if freeze_embeddings:
                self.embedding.weight.requires_grad = False

        self.dropout_in = nn.Dropout(dropout)

        # 2. Message Embedding Layer: Bidirectional GRU (hidden_size = 100 in both directions)
        self.bigru = nn.GRU(
            input_size=word_embed_size,
            hidden_size=mel_h_size,
            batch_first=True,
            bidirectional=True
        )

        # 3. Corpus Embedding Layer: Attention pooling across tweets for each day
        self.proj_u = nn.Linear(mel_h_size, mel_h_size, bias=False)
        self.w_u = nn.Linear(mel_h_size, 1, bias=False)
        self.dropout_ce = nn.Dropout(dropout)

    def forward(
        self,
        words: torch.Tensor,
        prices: torch.Tensor,
        n_words: Optional[torch.Tensor] = None,
        n_msgs: Optional[torch.Tensor] = None,
        ss_index: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        B, n_days, max_msgs, max_words = words.shape
        device = words.device
        total_msgs = B * n_days * max_msgs

        # Reshape words to process all messages: [B * 5 * 30, 40]
        flat_words = words.view(total_msgs, max_words)
        word_embed = self.dropout_in(self.embedding(flat_words)) # [total_msgs, 40, 50]

        # BiGRU encoding of tweet messages with batch chunking to avoid VRAM spikes on GPU
        chunk_size = 1024
        msg_embed_list = []
        flat_ss = ss_index.view(total_msgs).clamp(min=0, max=max_words - 1) if ss_index is not None else None
        flat_n_words = n_words.view(total_msgs) if n_words is not None else None

        for start_idx in range(0, total_msgs, chunk_size):
            end_idx = min(start_idx + chunk_size, total_msgs)
            cur_len = end_idx - start_idx
            chunk_embed = word_embed[start_idx:end_idx]

            gru_out, _ = self.bigru(chunk_embed) # [cur_len, 40, 200]
            out_f = gru_out[:, :, :self.mel_h_size]
            out_b = gru_out[:, :, self.mel_h_size:]

            if flat_ss is not None:
                cur_ss = flat_ss[start_idx:end_idx]
                chunk_indices = torch.arange(cur_len, device=device)
                m_f = out_f[chunk_indices, cur_ss]
                m_b = out_b[chunk_indices, cur_ss]
                msg_embed_list.append((m_f + m_b) / 2.0)
            else:
                mel_h = (out_f + out_b) / 2.0
                if flat_n_words is not None:
                    cur_nw = flat_n_words[start_idx:end_idx]
                    w_mask = (torch.arange(max_words, device=device).unsqueeze(0) < cur_nw.unsqueeze(1)).unsqueeze(-1)
                    denom = cur_nw.clamp(min=1).unsqueeze(1).float()
                    msg_embed_list.append((mel_h * w_mask.float()).sum(dim=1) / denom)
                else:
                    msg_embed_list.append(mel_h.mean(dim=1))

        msg_embed = torch.cat(msg_embed_list, dim=0) # [total_msgs, 100]

        # Reshape back to [B, 5, 30, 100]
        msg_embed = msg_embed.view(B, n_days, max_msgs, self.mel_h_size)

        # 4. Attention Pooling over the messages of each day (Corpus Embedding Layer)
        # u = dot(tanh(W * msg), w_u) -> [B, 5, 30]
        proj = torch.tanh(self.proj_u(msg_embed)) # [B, 5, 30, 100]
        u_score = self.w_u(proj).squeeze(-1)       # [B, 5, 30]

        # Mask empty message slots
        if n_msgs is not None:
            msg_mask = (torch.arange(max_msgs, device=device).unsqueeze(0).unsqueeze(0) < n_msgs.unsqueeze(-1)) # [B, 5, 30]
            u_score = u_score.masked_fill(~msg_mask, -1e9)

        attn_weights = torch.softmax(u_score, dim=-1) # [B, 5, 30]
        attn_weights = torch.nan_to_num(attn_weights, nan=0.0)

        # Daily corpus tweet representation: c_t = sum(attn * msg) -> [B, 5, 100]
        c_t = torch.matmul(attn_weights.unsqueeze(-2), msg_embed).squeeze(-2)
        c_t = self.dropout_ce(c_t)

        # 5. Price Feature Concatenation: [c_t (100d); prices (3d)] -> x_t (103d)
        x_t = torch.cat([c_t, prices], dim=-1) # [B, 5, 103]

        return x_t, c_t
