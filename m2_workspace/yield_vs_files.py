import pandas as pd
from pathlib import Path

TWEET_DIR = Path(r"D:\DOWNLOADS\stocknet\stocknet-dataset\tweet\preprocessed")

d = pd.read_csv(r"m2_workspace\rejection_details.csv")

scol = next(c for c in d.columns if "stock" in c.lower())
rcol = next(c for c in d.columns if "reason" in c.lower() or "status" in c.lower())

excluded = [
    "PRICE_MOVEMENT_FILTER",
    "PRICE_TIMELINE_INVALID",
    "PRICE_DATE_MISSING"
]

ok = d[~d[rcol].isin(excluded)]

t = ok.groupby(scol)[rcol].value_counts().unstack(fill_value=0)

t["yield"] = t.get("VALID_SAMPLE", 0) / t.sum(axis=1)

def n_files(stock):
    p = TWEET_DIR / str(stock)

    if not p.is_dir():
        return -1

    return sum(
        1 for f in p.iterdir()
        if "2014-01-01" <= f.name <= "2015-12-31"
    )

t["tweet_files"] = [n_files(s) for s in t.index]

cols = [
    "tweet_files",
    "VALID_SAMPLE",
    "TWEET_ALIGNMENT_FAILED",
    "yield"
]

print("\n========== YIELD VS TWEET FILES ==========")
print(t.sort_values("tweet_files", ascending=False)[cols].to_string())

print("\n========== MISSING TWEET DIRECTORIES ==========")
print([s for s in t.index if n_files(s) == -1])

print("\n========== MANY FILES BUT LOW YIELD ==========")
print(
    t[(t.tweet_files >= 250) & (t["yield"] < 0.30)].index.tolist()
)
