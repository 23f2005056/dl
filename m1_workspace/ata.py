import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

class AttentiveTemporalAuxiliary(nn.Module):
    """
    Attentive Temporal Auxiliary (ATA) module for StockNet.
    Corresponds exactly to original Model.py:596-621:
      - Independent day score: v_i = dot(tanh(proj_i(g)), w_i)
      - Dependent target similarity: v_d = dot(tanh(proj_d(g)), g_T)
      - Dual attention score: aux_score = v_i * v_d
      - Causal masking: strictly masks out target day (T-1) and padded days (t >= T)
      - Attention weights: v_stared = softmax(masked_aux_score)
      - Daily context aggregation: context = sum(v_stared * y) if daily_att == 'y' else sum(v_stared * g)
      - Target day prediction: y_T = Linear([context; g_T]) -> logits [B, 2]

    Inputs:
      H: [B, 5, 50] (temporal state sequence from VMD)
      T_lengths: [B] (true number of trading days, variable between 2 and 5)

    Outputs:
      logits: [B, 2] (final classification logits: fall/rise)
      attn_weights: [B, 5] (temporal attention weights strictly over past days)
      aux_logits: [B, 5, 2] (intermediate day movement predictions for auxiliary loss)
    """
    def __init__(
        self,
        hidden_dim: int = 50,
        num_classes: int = 2,
        alpha_aux: float = 0.5,
        daily_att: str = "y"
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.alpha_aux = alpha_aux
        self.daily_att = daily_att

        # 1. Independent day projection: proj_i(g), w_i (Model.py:597-600, no bias)
        self.proj_i = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.w_i = nn.Linear(hidden_dim, 1, bias=False)

        # 2. Dependent similarity projection: proj_d(g) (Model.py:601-603, no bias)
        self.proj_d = nn.Linear(hidden_dim, hidden_dim, bias=False)

        # 3. Auxiliary intermediate day classifier: g -> y [2d] (Model.py:570)
        self.aux_classifier = nn.Linear(hidden_dim, num_classes)

        # 4. Main prediction head: [context; g_T] -> logits [2d] (Model.py:620)
        context_dim = num_classes if daily_att == "y" else hidden_dim
        self.main_classifier = nn.Linear(context_dim + hidden_dim, num_classes)

    def forward(
        self,
        H: torch.Tensor,
        T_lengths: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, max_days, D = H.shape
        device = H.device

        # Step 1: Extract target day temporal state g_T (index T_lengths - 1)
        target_idx = (T_lengths - 1).clamp(min=0, max=max_days - 1) # [B]
        batch_idx = torch.arange(B, device=device)
        g_T = H[batch_idx, target_idx] # [B, 50]

        # Step 2: Auxiliary intermediate day predictions
        aux_logits = self.aux_classifier(H)          # [B, 5, 2]
        y_prob = torch.softmax(aux_logits, dim=-1)   # [B, 5, 2]

        # Step 3: Dual Temporal Attention Scoring (Model.py:596-607)
        # v_i = dot(tanh(proj_i(g)), w_i) -> [B, 5]
        p_i = torch.tanh(self.proj_i(H))
        v_i = self.w_i(p_i).squeeze(-1) # [B, 5]

        # v_d = dot(tanh(proj_d(g)), g_T) -> [B, 5]
        p_d = torch.tanh(self.proj_d(H))
        v_d = torch.sum(p_d * g_T.unsqueeze(1), dim=-1) # [B, 5]

        # aux_score = v_i * v_d (element-wise product)
        aux_score = v_i * v_d # [B, 5]

        # Step 4: Strict Causal Masking
        # Valid ONLY for intermediate days strictly before target day (t < T - 1)
        day_indices = torch.arange(max_days, device=device).unsqueeze(0).expand(B, max_days)
        valid_past_mask = (day_indices < target_idx.unsqueeze(1)) # [B, 5]

        masked_score = aux_score.masked_fill(~valid_past_mask, -1e9)
        attn_weights = torch.softmax(masked_score, dim=-1) # [B, 5]
        attn_weights = torch.nan_to_num(attn_weights, nan=0.0)
        attn_weights = attn_weights.masked_fill(~valid_past_mask, 0.0)

        # Step 5: Aggregate Daily Context (Model.py:612-618)
        if self.daily_att == "y":
            # Context is weighted sum of past movement predictions: [B, 2]
            att_c = torch.sum(attn_weights.unsqueeze(-1) * y_prob, dim=1)
        else:
            # Context is weighted sum of past hidden states: [B, 50]
            att_c = torch.sum(attn_weights.unsqueeze(-1) * H, dim=1)

        # Step 6: Final Prediction (Model.py:620)
        combined = torch.cat([att_c, g_T], dim=-1) # [B, 52] or [B, 100]
        logits = self.main_classifier(combined)      # [B, 2]

        return logits, attn_weights, aux_logits

    def compute_aux_loss(
        self,
        aux_logits: torch.Tensor,
        y_historical: torch.Tensor,
        attn_weights: torch.Tensor,
        T_lengths: torch.Tensor
    ) -> torch.Tensor:
        """
        Computes attention-weighted auxiliary cross-entropy loss over valid past days.
        """
        B, max_days, num_classes = aux_logits.shape
        device = aux_logits.device

        target_idx = (T_lengths - 1).clamp(min=0, max=max_days - 1)
        day_indices = torch.arange(max_days, device=device).unsqueeze(0).expand(B, max_days)
        valid_past_mask = (day_indices < target_idx.unsqueeze(1))

        if y_historical.dim() == 3 and y_historical.size(-1) == num_classes:
            y_targets = y_historical.argmax(dim=-1)
        else:
            y_targets = y_historical.long()

        loss_raw = F.cross_entropy(
            aux_logits.view(-1, num_classes),
            y_targets.view(-1),
            reduction="none"
        ).view(B, max_days)

        weighted_loss = loss_raw * attn_weights
        masked_loss = weighted_loss.masked_fill(~valid_past_mask, 0.0)
        return masked_loss.sum(dim=-1).mean()
