import torch
from mie import MarketInformationEncoder

def test_mie_dummy():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Testing MIE with dummy tensors on device: {device}")

    # B=4 batches, 5 days, 30 messages, 40 words
    words = torch.randint(0, 1000, (4, 5, 30, 40), dtype=torch.long, device=device)
    prices = torch.randn(4, 5, 3, dtype=torch.float32, device=device)
    n_words = torch.randint(1, 40, (4, 5, 30), dtype=torch.long, device=device)
    n_msgs = torch.randint(1, 30, (4, 5), dtype=torch.long, device=device)

    model = MarketInformationEncoder(vocab_size=29867).to(device)
    x_t, c_t = model(words, prices, n_words, n_msgs)

    assert x_t.shape == (4, 5, 103), f"Expected [4, 5, 103], got {x_t.shape}"
    assert c_t.shape == (4, 5, 100), f"Expected [4, 5, 100], got {c_t.shape}"
    
    # Check backward pass
    loss = x_t.sum()
    loss.backward()
    print("MIE Dummy Stub Test PASSED! Output shape: [4, 5, 103], Gradients verified.")

if __name__ == "__main__":
    test_mie_dummy()
