import torch
import torch.nn as nn
from mie import MarketInformationEncoder
from ata import AttentiveTemporalAuxiliary

class StockNet(nn.Module):
    """
    Complete StockNet Model Container.
    Connects MIE (M1) -> VMD (M4) -> ATA (M1).
    M1 owns the VMD -> ATA integration mapping here.
    """
    def __init__(self, vmd_module: nn.Module = None, vocab_size: int = 29867):
        super().__init__()
        # 1. Market Information Encoder (M1)
        self.mie = MarketInformationEncoder(vocab_size=vocab_size)

        # 2. Variational Movement Decoder (M4)
        self.vmd = vmd_module

        # 3. Attentive Temporal Auxiliary (M1)
        self.ata = AttentiveTemporalAuxiliary(hidden_dim=50, num_classes=2)

    def forward(self, batch: dict):
        """
        End-to-End Forward Pass:
          batch -> MIE -> VMD -> ATA -> (logits, total_loss, aux_logits, attn_weights)
        """
        # Step A: MIE encodes tweet text + prices -> x_t: [B, 5, 103]
        x_t, c_t = self.mie(
            words=batch["tweet_ids"],
            prices=batch["price_features"],
            n_words=batch["n_words"],
            n_msgs=batch["n_msgs"]
        )

        # Step B: VMD processes x_t and samples latent z_t -> H: [B, 5, 50]
        if self.vmd is not None:
            H, kl_loss = self.vmd(x_t, batch["T_lengths"], batch.get("y_historical"))
        else:
            # Fallback if VMD is being tested standalone
            H = torch.randn(x_t.size(0), x_t.size(1), 50, device=x_t.device)
            kl_loss = torch.tensor(0.0, device=x_t.device)

        # Step C: VMD -> ATA Mapping (M1 OWNS THIS STEP)
        # Passes VMD state sequence H [B, 5, 50] into ATA with sequence lengths T
        logits, attn_weights, aux_logits = self.ata(H, batch["T_lengths"])

        return logits, kl_loss, aux_logits, attn_weights
