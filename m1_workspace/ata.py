import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentiveTemporalAuxiliary(nn.Module):
    """
    Attentive Temporal Auxiliary (ATA) for StockNet.
    Corresponds to Model.py:
      - Temporal Attention: computes attention weights over past trading days (0 .. T-2)
      - Causal Masking: masks out padded days (t >= T) and current target day (t = T-1)
      - Context Aggregation: weighted sum of past hidden states
      - Main Classifier: predicts target day stock movement (0=fall, 1=rise)
      - Auxiliary Classifier: intermediate loss to prevent vanishing gradients

    Input:
      H:         [B, 5, 50] (Hidden states from VMD)
      T_lengths: [B]        (Actual trading days per sample, values 2 to 5)

    Output:
      logits:     [B, 2]     (Target day class prediction logits)
      attn_weights: [B, 5]   (Attention weights across days)
      aux_logits: [B, 5, 2]  (Auxiliary intermediate logits)
    """
    def __init__(
        self,
        hidden_dim: int = 50,
        num_classes: int = 2,
        alpha_aux: float = 0.5
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.alpha_aux = alpha_aux

        # Temporal attention projections
        self.proj_i = nn.Linear(hidden_dim, hidden_dim)
        self.proj_T = nn.Linear(hidden_dim, hidden_dim)
        self.v_attn = nn.Linear(hidden_dim, 1, bias=False)

        # Main prediction head from [context; h_T] -> 2 classes
        self.main_classifier = nn.Linear(hidden_dim * 2, num_classes)

        # Auxiliary classifier head: predicts movement at each intermediate step
        self.aux_classifier = nn.Linear(hidden_dim, num_classes)

    def forward(
        self,
        H: torch.Tensor,
        T_lengths: torch.Tensor
    ):
        B, max_days, D = H.shape
        device = H.device

        # 1. Extract target day hidden state h_{T-1} for each sample
        # T_lengths has values 2..5, so target index is T_lengths - 1
        target_idx = (T_lengths - 1).clamp(min=0, max=max_days - 1) # [B]
        batch_idx = torch.arange(B, device=device)
        h_T = H[batch_idx, target_idx] # [B, 50]

        # 2. Compute temporal attention scores between intermediate days and target day
        # score_t = v^T * tanh(W_i * h_t + W_T * h_T)
        h_i_proj = self.proj_i(H) # [B, 5, 50]
        h_T_proj = self.proj_T(h_T).unsqueeze(1) # [B, 1, 50]
        scores = self.v_attn(torch.tanh(h_i_proj + h_T_proj)).squeeze(-1) # [B, 5]

        # 3. Create Causal Mask:
        # A day t is valid for attention ONLY if t < target_idx (i.e. strictly before target day)
        day_indices = torch.arange(max_days, device=device).unsqueeze(0).expand(B, max_days)
        valid_past_mask = (day_indices < target_idx.unsqueeze(1)) # [B, 5] boolean mask

        # Mask invalid days (target day and padded days) with -1e9
        scores = scores.masked_fill(~valid_past_mask, -1e9)
        attn_weights = torch.softmax(scores, dim=-1) # [B, 5]
        # Replace NaNs (if no past day exists, though T >= 2 guarantees at least 1)
        attn_weights = torch.nan_to_num(attn_weights, nan=0.0)

        # 4. Context Vector: weighted sum of past hidden states
        context = torch.sum(H * attn_weights.unsqueeze(-1), dim=1) # [B, 50]

        # 5. Main Prediction: concatenate context and target day hidden state [c; h_T]
        combined = torch.cat([context, h_T], dim=-1) # [B, 100]
        logits = self.main_classifier(combined)      # [B, 2]

        # 6. Auxiliary Predictions for intermediate steps
        aux_logits = self.aux_classifier(H) # [B, 5, 2]

        return logits, attn_weights, aux_logits

    def compute_aux_loss(
        self,
        aux_logits: torch.Tensor,
        y_historical: torch.Tensor,
        attn_weights: torch.Tensor,
        T_lengths: torch.Tensor
    ) -> torch.Tensor:
        """
        Computes attention-weighted auxiliary cross-entropy loss over intermediate days.
        y_historical: [B, 5, 2] one-hot labels
        """
        B, max_days, num_classes = aux_logits.shape
        device = aux_logits.device

        # Cross-entropy over all steps
        flat_logits = aux_logits.view(-1, num_classes)
        flat_targets = y_historical.view(-1, num_classes).argmax(dim=-1)
        ce_per_step = F.cross_entropy(flat_logits, flat_targets, reduction='none').view(B, max_days)

        # Weight by temporal attention alpha_t
        weighted_ce = ce_per_step * attn_weights

        # Sum over steps and average over batch
        return weighted_ce.sum(dim=-1).mean()
