import torch
import torch.nn as nn
import torch.nn.functional as F

class MarketInformationEncoder(nn.Module):
    """
    Market Information Encoder (MIE) for StockNet.
    Corresponds to original Model.py:
      - Word Embedding: [vocab_size=29867, 50d]
      - Message Embedding Layer (BiGRU): forward + backward average -> 100d
      - Corpus Embedding Layer (Attention Pooling over tweets): -> c_t [100d]
      - Price Concatenation: [c_t (100d); p_t (3d)] -> x_t [103d]

    Input Shapes:
      words:  [B, 5, 30, 40] (int64 token IDs)
      prices: [B, 5, 3]      (float32 normalized [H, L, C])
      n_words: [B, 5, 30]    (int64 actual words per message)
      n_msgs:  [B, 5]        (int64 actual tweets per day)

    Output Shapes:
      x: [B, 5, 103] -> Feeds directly into M4's VMD recurrent cell
      c: [B, 5, 100] -> Tweet-only daily representation
    """
    def __init__(
        self,
        vocab_size: int = 29867,
        word_embed_size: int = 50,
        mel_h_size: int = 100,
        price_dim: int = 3,
        dropout: float = 0.3
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.word_embed_size = word_embed_size
        self.mel_h_size = mel_h_size
        self.price_dim = price_dim

        # 1. Word Embedding (padding_idx=0 handles PAD and UNK)
        self.embedding = nn.Embedding(vocab_size, word_embed_size, padding_idx=0)
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
        n_words: torch.Tensor = None,
        n_msgs: torch.Tensor = None
    ) -> torch.Tensor:
        B, n_days, max_msgs, max_words = words.shape
        device = words.device

        # Reshape words to process all messages in parallel: [B * 5 * 30, 40]
        flat_words = words.view(B * n_days * max_msgs, max_words)
        word_embed = self.dropout_in(self.embedding(flat_words)) # [B*5*30, 40, 50]

        # BiGRU encoding of each tweet message
        gru_out, _ = self.bigru(word_embed) # [B*5*30, 40, 200]
        # Average forward and backward representations across words: -> [B*5*30, 100]
        out_f = gru_out[:, :, :self.mel_h_size]
        out_b = gru_out[:, :, self.mel_h_size:]
        mel_h = (out_f + out_b) / 2.0 # [B*5*30, 40, 100]

        # Extract message vector at last valid word or stock-token index (mean over valid words)
        if n_words is not None:
            flat_n_words = n_words.view(B * n_days * max_msgs)
            # Create word mask: True for valid tokens, False for padding
            word_mask = (torch.arange(max_words, device=device).unsqueeze(0) < flat_n_words.unsqueeze(1)).unsqueeze(-1)
            masked_mel = mel_h * word_mask.float()
            denom = flat_n_words.clamp(min=1).unsqueeze(1).float()
            msg_embed = masked_mel.sum(dim=1) / denom # [B*5*30, 100]
        else:
            msg_embed = mel_h.mean(dim=1)

        # Reshape back to [B, 5, 30, 100]
        msg_embed = msg_embed.view(B, n_days, max_msgs, self.mel_h_size)

        # 4. Attention Pooling over the messages of each day
        # u = dot(tanh(W * msg), w_u) -> [B, 5, 30]
        proj = torch.tanh(self.proj_u(msg_embed)) # [B, 5, 30, 100]
        u_score = self.w_u(proj).squeeze(-1)       # [B, 5, 30]

        # Mask empty message slots
        if n_msgs is not None:
            msg_mask = (torch.arange(max_msgs, device=device).unsqueeze(0).unsqueeze(0) < n_msgs.unsqueeze(-1)) # [B, 5, 30]
            u_score = u_score.masked_fill(~msg_mask, -1e9)

        attn_weights = torch.softmax(u_score, dim=-1) # [B, 5, 30]
        # Replace NaNs (days with 0 messages) with 0
        attn_weights = torch.nan_to_num(attn_weights, nan=0.0)

        # Daily corpus tweet representation: c_t = sum(attn * msg) -> [B, 5, 100]
        c_t = torch.matmul(attn_weights.unsqueeze(-2), msg_embed).squeeze(-2)
        c_t = self.dropout_ce(c_t)

        # 5. Price Feature Concatenation: [c_t (100d); prices (3d)] -> x_t (103d)
        x_t = torch.cat([c_t, prices], dim=-1) # [B, 5, 103]

        return x_t, c_t
