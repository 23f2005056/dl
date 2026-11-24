import torch
from ata import AttentiveTemporalAuxiliary

def test_ata_dummy():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Testing ATA with dummy tensors on device: {device}")

    # B=4 samples, 5 days, hidden_dim=50
    H = torch.randn(4, 5, 50, dtype=torch.float32, device=device)
    # Variable T: sample 0 has T=3, sample 1 has T=4, sample 2 has T=5, sample 3 has T=2
    T_lengths = torch.tensor([3, 4, 5, 2], dtype=torch.long, device=device)
    y_hist = torch.zeros(4, 5, 2, dtype=torch.float32, device=device)
    y_hist[:, :, 0] = 1.0 # dummy one-hot labels

    ata = AttentiveTemporalAuxiliary(hidden_dim=50, num_classes=2).to(device)
    logits, attn_weights, aux_logits = ata(H, T_lengths)

    assert logits.shape == (4, 2), f"Expected [4, 2], got {logits.shape}"
    assert attn_weights.shape == (4, 5), f"Expected [4, 5], got {attn_weights.shape}"
    assert aux_logits.shape == (4, 5, 2), f"Expected [4, 5, 2], got {aux_logits.shape}"

    # Verify causal masking: Target day (T-1) and padded days (>=T) must have 0 attention weight
    for b in range(4):
        T = T_lengths[b].item()
        target_idx = T - 1
        # Target day attention must be 0
        assert attn_weights[b, target_idx].item() == 0.0, f"Sample {b}: target day attention not 0!"
        # Padded days attention must be 0
        for pad_t in range(T, 5):
            assert attn_weights[b, pad_t].item() == 0.0, f"Sample {b}: padded day {pad_t} attention not 0!"
        # Past days attention must sum to 1.0
        sum_past = attn_weights[b, :target_idx].sum().item()
        assert abs(sum_past - 1.0) < 1e-4, f"Sample {b}: past attention does not sum to 1 (got {sum_past})!"

    aux_loss = ata.compute_aux_loss(aux_logits, y_hist, attn_weights, T_lengths)
    total_loss = logits.sum() + aux_loss
    total_loss.backward()

    print("ATA Dummy Stub Test PASSED! Logits: [4, 2], Causal masking strictly verified, Gradients OK.")

if __name__ == "__main__":
    test_ata_dummy()
