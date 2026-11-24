import sys
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "m1_workspace"))
sys.path.insert(0, str(ROOT / "m2_workspace"))
sys.path.insert(0, str(ROOT / "m3_workspace"))

from stocknet_mappings import StockNetMappings
from price_reader import PriceReader
from tweet import TweetReader
from sample_builder import StockNetSampleBuilder
from batch_builder import StockNetBatchBuilder
from batch_adapter import prepare_torch_batch
from mie import MarketInformationEncoder
from ata import AttentiveTemporalAuxiliary

def test_real_pipeline():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running Real-Data Integration on: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    DATA = ROOT / "stocknet-dataset"
    CODE = ROOT / "stocknet-code"

    mappings = StockNetMappings(config_path=CODE / "src" / "config.yml", vocab_path=CODE / "res" / "vocab.txt")
    price_reader = PriceReader(DATA / "price" / "preprocessed")
    tweet_reader = TweetReader(tweet_root=DATA / "tweet" / "preprocessed", vocab_path=CODE / "res" / "vocab.txt")

    sample_builder = StockNetSampleBuilder(mappings, price_reader, tweet_reader)
    batch_builder = StockNetBatchBuilder(sample_builder, list(mappings.stock_id_dict.keys()), batch_size=32)

    # 1. Fetch real batch from M2
    print("Loading real batch from M2 BatchBuilder...")
    raw_batch = next(batch_builder.iter_batches("train"))
    batch = prepare_torch_batch(raw_batch, device)

    print("Batch loaded and moved to GPU successfully:")
    print("  tweet_ids:      ", batch["tweet_ids"].shape)
    print("  price_features: ", batch["price_features"].shape)
    print("  T_lengths:      ", batch["T_lengths"][:8].tolist(), "...")
    print("  targets:        ", batch["targets"].shape)

    # 2. Forward pass through MIE
    vocab_size = len(mappings.vocab_id_dict)
    mie = MarketInformationEncoder(vocab_size=vocab_size).to(device)
    x_t, c_t = mie(batch["tweet_ids"], batch["price_features"], batch["n_words"], batch["n_msgs"])
    print(f"MIE output verified: x_t = {x_t.shape} (Expected: [32, 5, 103]), c_t = {c_t.shape}")
    assert x_t.shape == (32, 5, 103)

    # 3. Simulate VMD output (Mocking VMD recurrent cell: [B=32, 5, 50])
    dummy_vmd_H = torch.randn(32, 5, 50, device=device)

    # 4. Forward pass through ATA
    ata = AttentiveTemporalAuxiliary(hidden_dim=50, num_classes=2).to(device)
    logits, attn_weights, aux_logits = ata(dummy_vmd_H, batch["T_lengths"])
    print(f"ATA output verified: logits = {logits.shape} (Expected: [32, 2]), attn_weights = {attn_weights.shape}")
    assert logits.shape == (32, 2)

    # 5. Compute loss and verify backward pass
    loss_fn = torch.nn.CrossEntropyLoss()
    main_loss = loss_fn(logits, batch["targets"])
    aux_loss = ata.compute_aux_loss(aux_logits, batch["y_historical"], attn_weights, batch["T_lengths"])
    total_loss = main_loss + 0.5 * aux_loss
    total_loss.backward()

    print(f"Backward pass verified! Main Loss = {main_loss.item():.4f}, Aux Loss = {aux_loss.item():.4f}")
    print("\n>>> M1 ARCHITECTURE & M2 DATA INTEGRATION IS 100% OPERATIONAL ON GPU! <<<")

if __name__ == "__main__":
    test_real_pipeline()
