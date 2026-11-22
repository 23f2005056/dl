"""Audit all original preprocessed date files using the M3 handoff reader."""
import json
from collections import Counter

from tweet import MAX_MESSAGES, MAX_WORDS, PROJECT, TWEETS, TweetReader


def main():
    reader = TweetReader()
    by_stock = Counter()
    files = messages = unknown = truncated_files = truncated_messages = 0
    empty_files = 0
    parser_files_checked = 0
    lowest_id, highest_id = reader.vocab_size, 0
    for stock_dir in sorted(TWEETS.iterdir()):
        if not stock_dir.is_dir():
            continue
        for path in sorted(stock_dir.iterdir()):
            if not path.is_file():
                continue
            files += 1
            by_stock[stock_dir.name] += 1
            with path.open(encoding="utf-8") as stream:
                raw = [json.loads(line)["text"] for line in stream]
            usable = [tokens for tokens in raw if tokens]
            if not usable:
                empty_files += 1
            if len(usable) > MAX_MESSAGES:
                truncated_files += 1
            truncated_messages += sum(len(tokens) > MAX_WORDS for tokens in usable[:MAX_MESSAGES])
            selected = [(line, tokens[:MAX_WORDS]) for line, tokens in enumerate(raw, 1) if tokens][:MAX_MESSAGES]
            # A periodic full-reader comparison exercises the public handoff
            # without reopening every tiny file on slow filesystems.
            if files == 1 or files % 250 == 0:
                record = reader.get(stock_dir.name, path.name)
                assert record["n_msgs"] == len(selected)
                assert [(m["source_line"], m["word_ids"], m["n_words"], m["ss_index"])
                        for m in record["messages"]] == [
                            (line, [reader.token_to_id.get(w, 0) for w in tokens], len(tokens),
                             reader.stock_index(tokens, stock_dir.name)) for line, tokens in selected]
                parser_files_checked += 1
            for _, tokens in selected:
                ids = [reader.token_to_id.get(w, 0) for w in tokens]
                assert len(ids) <= MAX_WORDS
                assert 0 <= reader.stock_index(tokens, stock_dir.name) < len(ids)
                assert all(0 <= value < reader.vocab_size for value in ids)
                messages += 1
                unknown += ids.count(0)
                if ids:
                    lowest_id = min(lowest_id, min(ids))
                    highest_id = max(highest_id, max(ids))
    price_stocks = {p.stem for p in (PROJECT / "stocknet-dataset/price/preprocessed").glob("*.txt")}
    print(json.dumps({"tweet_stocks": len(by_stock), "price_stocks": len(price_stocks),
                      "price_stocks_without_tweets": sorted(price_stocks - by_stock.keys()),
                      "date_files": files, "empty_date_files": empty_files,
                      "files_over_30_usable_messages": truncated_files,
                      "messages_handed_off": messages,
                      "parser_files_cross_checked": parser_files_checked,
                      "messages_over_40_words_before_clipping": truncated_messages,
                      "unknown_ids_in_handoff": unknown,
                      "id_min": lowest_id, "id_max": highest_id,
                      "vocab_size_including_unknown": reader.vocab_size,
                      "sparse_stock_file_counts": {s: by_stock[s] for s in ("GMRE", "AGFS", "SPLP", "PICO", "AAPL")}}, indent=2))


if __name__ == "__main__":
    main()
