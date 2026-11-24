import csv

FIELDS = ["split", "stock", "target_date", "y_true", "y_pred", "p_rise", "checkpoint"]


def write_csv(path, split, rows, checkpoint):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow({**r, "split": split, "checkpoint": checkpoint})