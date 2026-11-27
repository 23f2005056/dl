import pandas as pd
d = pd.read_csv(r"m2_workspace\rejection_details.csv")
scol = next(c for c in d.columns if "stock" in c.lower())
rcol = next(c for c in d.columns if "reason" in c.lower() or "status" in c.lower())
price_ok = d[~d[rcol].isin(["PRICE_MOVEMENT_FILTER", "PRICE_TIMELINE_INVALID", "PRICE_DATE_MISSING"])]
t = price_ok.groupby(scol)[rcol].value_counts().unstack(fill_value=0)
t["yield"] = t.get("VALID_SAMPLE", 0) / t.sum(axis=1)
print(t.sort_values("yield", ascending=False).to_string())