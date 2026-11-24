import datetime
import torch
from torch.utils.data import Dataset
from batch import MAX_DAYS, MAX_MSGS, MAX_WORDS, PRICE_DIM, VOCAB_SIZE

START = {"train": datetime.date(2014, 1, 1), "dev": datetime.date(2015, 8, 1),
         "test": datetime.date(2015, 10, 1)}


class FakeDataset(Dataset):
    def __init__(self, n, seed, split):
        g = torch.Generator().manual_seed(seed)
        self.samples = []
        for i in range(n):
            T = int(torch.randint(2, MAX_DAYS + 1, (1,), generator=g))
            n_msgs = torch.zeros(MAX_DAYS, dtype=torch.long)
            n_msgs[:T] = torch.randint(1, MAX_MSGS + 1, (T,), generator=g)
            msg_mask = torch.arange(MAX_MSGS)[None, :] < n_msgs[:, None]
            n_words = torch.randint(1, MAX_WORDS + 1, (MAX_DAYS, MAX_MSGS), generator=g) * msg_mask
            word = torch.randint(1, VOCAB_SIZE, (MAX_DAYS, MAX_MSGS, MAX_WORDS), generator=g)
            word = word * (torch.arange(MAX_WORDS) < n_words[..., None])
            price = torch.zeros(MAX_DAYS, PRICE_DIM)
            price[:T] = torch.randn(T, PRICE_DIM, generator=g) * 0.01
            mv = price[:, 2].clone()
            y = torch.zeros(MAX_DAYS, 2)
            y[torch.arange(T), (mv[:T] > 0).long()] = 1.0
            self.samples.append({
                "stock": 0, "T": T, "word": word, "n_words": n_words, "n_msgs": n_msgs,
                "ss_index": torch.zeros(MAX_DAYS, MAX_MSGS, dtype=torch.long),
                "price": price, "mv_percent": mv, "y": y,
                "stock_symbol": "FAKE",
                "target_date": (START[split] + datetime.timedelta(days=i)).isoformat(),
            })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        return self.samples[i]