
import os
import sys
import io
import json
import types
import logging
import argparse
import itertools
import time
from pathlib import Path

import numpy as np
import yaml


# ============================================================
# 1. PROJECT PATHS
# ============================================================

ROOT = Path(
    os.environ.get("STOCKNET_ROOT", r"D:\DOWNLOADS\stocknet")
)

CODE_SRC = ROOT / "stocknet-code" / "src"
DATASET = ROOT / "stocknet-dataset"
VOCAB = ROOT / "stocknet-code" / "res" / "vocab.txt"

sys.path.insert(0, str(ROOT / "m2_workspace"))
sys.path.insert(0, str(ROOT / "m3_workspace"))


# ============================================================
# 2. LOAD ORIGINAL DATAPIPE UNDER PYTHON 3
# ============================================================

cfg = yaml.safe_load(
    open(CODE_SRC / "config.yml", encoding="utf-8")
)

config_model = dict(cfg["model"])
config_model["shuffle"] = 0

stock_symbols = list(
    itertools.chain.from_iterable(cfg["stocks"].values())
)

vocab = json.loads(
    VOCAB.read_text(encoding="utf-8")
)


# Replace original hardcoded data paths.
class _Paths:
    movement = str(DATASET / "price" / "preprocessed")
    preprocessed = str(DATASET / "tweet" / "preprocessed")
    vocab = str(VOCAB)
    glove = ""


# Provide the ConfigLoader module expected by DataPipe.py.
stub = types.ModuleType("ConfigLoader")

stub.logger = logging.getLogger("oracle")
stub.path_parser = _Paths
stub.config_model = config_model
stub.dates = cfg["dates"]
stub.stock_symbols = stock_symbols
stub.vocab = vocab
stub.vocab_size = len(vocab) + 1

sys.modules["ConfigLoader"] = stub


# Read and execute original DataPipe.py in memory.
src = (CODE_SRC / "DataPipe.py").read_text(encoding="utf-8")

old = "with open(msg_fp, 'r') as tweet_f:"

assert old in src, (
    "Tweet-open line not found. "
    "Your original DataPipe.py may differ from the reviewed version."
)

# Python 3 encoding compatibility adjustment only.
src = src.replace(
    old,
    "with io.open(msg_fp, 'r', encoding='utf8') as tweet_f:"
)

ns = {"__name__": "original_datapipe"}

exec(
    compile(src, "DataPipe.py", "exec"),
    ns
)

pipe = ns["DataPipe"]()

vocab_id_dict = pipe.index_token(vocab, key="token")

stock_id_dict = pipe.index_token(
    stock_symbols,
    key="token",
    type="stock"
)

# Reverse mappings allow comparison by stock symbol.
original_id_to_symbol = {
    value: key for key, value in stock_id_dict.items()
}


# ============================================================
# 3. INITIALIZE YOUR M2 PIPELINE
# ============================================================

from oracle_m2_factory import make_builder

builder = make_builder(stock_symbols)

m2_id_to_symbol = {
    value: key
    for key, value in builder.mappings.stock_id_dict.items()
}


# ============================================================
# 4. COMPARISON FUNCTIONS
# ============================================================

def same(a, b, tol=None):

    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)

    if a.shape != b.shape:
        return False

    if tol is not None:
        return np.allclose(
            a, b,
            atol=tol,
            rtol=0
        )

    return np.array_equal(a, b)


def diff_fields(o, m):

    bad = []

    # Compare stock identity, not numerical ID.
    original_symbol = original_id_to_symbol.get(
        int(o["stock"])
    )

    m2_symbol = m2_id_to_symbol.get(
        int(m["stock"])
    )

    if original_symbol != m2_symbol:
        bad.append("stock_symbol")

    if int(o["T"]) != int(m["T"]):
        bad.append("T")

    if not same(o["ys"], m["ys"]):
        bad.append("ys")

    if abs(
        float(o["main_mv_percent"]) -
        float(m["main_mv_percent"])
    ) > 1e-9:
        bad.append("main_mv_percent")

    if not same(o["mv_percents"], m["mv_percents"]):
        bad.append("mv_percents")

    if not same(o["prices"], m["prices"], tol=1e-9):
        bad.append("prices")

    for k in ("msgs", "ss_indices", "n_words", "n_msgs"):

        if not same(o[k], m[k]):
            bad.append(k)

    return bad


# ============================================================
# 5. RUN COMPARISON
# ============================================================

def main():

    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--stocks",
        nargs="*",
        default=None
    )

    ap.add_argument(
        "--all",
        action="store_true"
    )

    ap.add_argument(
        "--phases",
        nargs="*",
        default=["train", "dev", "test"]
    )

    a = ap.parse_args()

    default_stocks = [
        "AAPL",
        "FB",
        "MSFT",
        "XOM",
        "KO",
        "REX",
        "BABA",
        "AGFS",
        "BRK-A",
        "GMRE"
    ]

    stocks = (
        stock_symbols
        if a.all
        else (a.stocks or default_stocks)
    )

    totals = {
        "orig": 0,
        "m2": 0,
        "only_orig": 0,
        "only_m2": 0,
        "mismatch": 0
    }

    problems = []

    t0 = time.time()

    print(
        f"{'stock':8}"
        f"{'phase':7}"
        f"{'orig':>6}"
        f"{'m2':>6}"
        f"{'only_orig':>10}"
        f"{'only_m2':>8}"
        f"{'mismatch':>9}"
        f"{'order':>7}"
    )

    for s in stocks:

        for ph in a.phases:

            # Original DataPipe samples.
            o = {
                x["main_target_date"]: x
                for x in pipe.sample_gen_from_one_stock(
                    vocab_id_dict,
                    stock_id_dict,
                    s,
                    ph
                )
            }

            # Your M2 samples.
            m = {
                x["main_target_date"]: x
                for x in builder.iter_samples(
                    s,
                    ph,
                    shuffle=False
                )
            }

            only_o = sorted(set(o) - set(m))
            only_m = sorted(set(m) - set(o))

            common_dates = sorted(set(o) & set(m))

            bad = {
                d: diff_fields(o[d], m[d])
                for d in common_dates
            }

            bad = {
                d: fields
                for d, fields in bad.items()
                if fields
            }

            totals["orig"] += len(o)
            totals["m2"] += len(m)
            totals["only_orig"] += len(only_o)
            totals["only_m2"] += len(only_m)
            totals["mismatch"] += len(bad)

            print(
                f"{s:8}"
                f"{ph:7}"
                f"{len(o):6}"
                f"{len(m):6}"
                f"{len(only_o):10}"
                f"{len(only_m):8}"
                f"{len(bad):9}"
                f"{str(list(o) == list(m)):>7}",
                flush=True
            )

            for d in only_o[:2]:
                problems.append(
                    (s, ph, d, "only in ORIGINAL")
                )

            for d in only_m[:2]:
                problems.append(
                    (s, ph, d, "only in M2")
                )

            for d, fields in list(bad.items())[:2]:
                problems.append(
                    (
                        s,
                        ph,
                        d,
                        "fields differ: " + ",".join(fields)
                    )
                )

    print("\nTOTALS:", totals, f"({time.time()-t0:.0f}s)")

    ok = (
        totals["only_orig"] == 0
        and totals["only_m2"] == 0
        and totals["mismatch"] == 0
    )

    print(
        "RESULT:",
        "PASS - M2 pipeline identical to original on these stocks"
        if ok
        else "FAIL"
    )

    for p in problems[:15]:
        print("  ", p)


if __name__ == "__main__":
    main()
