"""StockNet preprocessed tweet reader. Calendar alignment belongs to M2.

The original DataPipe clips each date file to 30 nonempty messages and each
message to 40 tokens. M2 must clip to 30 *again* after merging calendar dates
for one trading step, in ascending calendar date and file order.
"""
import json
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[1]
TWEETS = PROJECT / "stocknet-dataset/tweet/preprocessed"
VOCAB = PROJECT / "stocknet-code/res/vocab.txt"
MAX_MESSAGES = 30
MAX_WORDS = 40


class TweetReader:
    def __init__(self, tweet_root=TWEETS, vocab_path=VOCAB):
        self.tweet_root = Path(tweet_root)
        words = json.loads(Path(vocab_path).read_text(encoding="utf-8"))
        # DataPipe.index_token inserts UNK at index 0. Match duplicate-token
        # behavior too: its dict assignment leaves the last occurrence.
        self.token_to_id = {word: index for index, word in enumerate(words, 1)}
        self.vocab_size = len(words) + 1

    @staticmethod
    def stock_index(tokens, symbol):
        """Match DataPipe.get_ss_index on the clipped token sequence."""
        symbol = symbol.lower()
        index = len(tokens) - 1
        if symbol in tokens:
            return tokens.index(symbol)
        if "$" in tokens:
            dollar = tokens.index("$")
            if dollar != len(tokens) - 1 and symbol in tokens[dollar + 1]:
                return dollar + 1
            for pos in range(dollar + 1, len(tokens)):
                if symbol in tokens[pos]:
                    return pos
        return index

    def get(self, symbol, calendar_date):
        """Return one date's source-order messages, with no trading-day mapping.

        An absent file and an existing file with zero usable messages both
        yield zero messages; `file_exists` distinguishes them for auditing.
        """
        path = self.tweet_root / symbol / str(calendar_date)
        messages = []
        if path.is_file():
            with path.open(encoding="utf-8") as stream:
                for line_number, line in enumerate(stream, 1):
                    tokens = json.loads(line)["text"]
                    if not tokens:
                        continue
                    tokens = tokens[:MAX_WORDS]
                    messages.append({
                        "word_ids": [self.token_to_id.get(w, 0) for w in tokens],
                        "n_words": len(tokens),
                        "ss_index": self.stock_index(tokens, symbol),
                        "source_line": line_number,
                    })
                    if len(messages) == MAX_MESSAGES:
                        break
        return {
            "stock": symbol,
            "calendar_date": str(calendar_date),
            "file_exists": path.is_file(),
            "messages": messages,
            "n_msgs": len(messages),
        }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("stock")
    parser.add_argument("calendar_date")
    args = parser.parse_args()
    print(json.dumps(TweetReader().get(args.stock, args.calendar_date), indent=2))
