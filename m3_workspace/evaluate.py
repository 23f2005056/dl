"""Evaluate identity-preserving E0 prediction CSV from M4."""
import argparse
import csv
import json
import math
from datetime import date
from functools import lru_cache
from pathlib import Path

from tweet import PROJECT


SPLITS = {
    "dev": (date(2015, 8, 1), date(2015, 10, 1)),
    "test": (date(2015, 10, 1), date(2016, 1, 1)),
}


@lru_cache(maxsize=None)
def price_movements(stock):
    path = PROJECT / "stocknet-dataset/price/preprocessed" / (stock + ".txt")
    if not path.is_file():
        raise ValueError("missing price file for " + stock)
    result = {}
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            columns = line.split("\t")
            result[columns[0]] = float(columns[1])
    return result


def evaluate(rows, split, check_price_labels=True):
    if split not in SPLITS:
        raise ValueError("split must be dev or test")
    start, end = SPLITS[split]
    counts = {"tp": 0, "fp": 0, "tn": 0, "fn": 0}
    seen = set()
    for row in rows:
        if row["split"] != split:
            raise ValueError("mixed or incorrect split")
        key = (row["stock"], row["target_date"])
        if not key[0] or key in seen:
            raise ValueError("missing stock or duplicate stock/date")
        seen.add(key)
        day = date.fromisoformat(key[1])
        if not start <= day < end:
            raise ValueError("target date outside split: " + key[1])
        true, pred = int(row["y_true"]), int(row["y_pred"])
        if true not in (0, 1) or pred not in (0, 1):
            raise ValueError("labels must be 0 or 1")
        if check_price_labels:
            movement = price_movements(key[0]).get(key[1])
            if movement is None or -0.005 <= movement < 0.0055:
                raise ValueError("missing or dead-band target: " + str(key))
            if true != int(movement > 1e-7):
                raise ValueError("label disagrees with price movement: " + str(key))
        counts[{(1, 1): "tp", (0, 1): "fp", (0, 0): "tn", (1, 0): "fn"}[(true, pred)]] += 1
    n = sum(counts.values())
    if not n:
        raise ValueError("no predictions")
    tp, fp, tn, fn = (counts[k] for k in ("tp", "fp", "tn", "fn"))
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None
    denominator = (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
    mcc = (tp * tn - fp * fn) / math.sqrt(denominator) if denominator else None
    return {"split": split, "n": n, "positive_class": "rise (1)",
            "accuracy": (tp + tn) / n, "mcc": mcc,
            "precision": precision, "recall": recall, "f1": f1,
            "confusion_matrix": [[tn, fp], [fn, tp]],
            "confusion_labels": ["fall (0)", "rise (1)"]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("predictions_csv")
    parser.add_argument("--split", choices=SPLITS, required=True)
    args = parser.parse_args()
    with open(args.predictions_csv, newline="", encoding="utf-8") as stream:
        result = evaluate(csv.DictReader(stream), args.split)
    print(json.dumps(result, indent=2))
