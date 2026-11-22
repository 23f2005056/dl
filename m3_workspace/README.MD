# M3 E0 handoff

## Tweet records for M2

`tweet.py` reads already tokenized JSONL from `stocknet-dataset/tweet/preprocessed/<stock>/<calendar-date>` using `stocknet-code/res/vocab.txt`.

```python
from tweet import TweetReader
record = TweetReader().get("AAPL", "2014-01-02")
first = record["messages"][0]
```

`record` has `stock`, `calendar_date`, `file_exists`, `messages`, `n_msgs`.
Each message has `word_ids` (integer list), `n_words`, `ss_index` (zero-based
stock-token position), and `source_line` (one-based JSONL line). Missing and
empty files both have zero messages; `file_exists` distinguishes them. No
calendar-to-trading assignment is performed here.

The first AAPL 2014-01-02 tweet is `['$', 'aapl', '-', 'wall', 'st', '.',
'kicks', 'off', 'new', 'year', 'on', 'lower', 'note', '->', 'URL', 'stock',
'stocks', 'stockaction']`. Its IDs are `[10377, 7341, 13080, 22140, 3353,
322, 10817, 1714, 11044, 2435, 13776, 14182, 22974, 17567, 8197,
27605, 28881, 25473]`, `n_words=18`, `ss_index=1`, `source_line=1`.
Vocabulary positions are one-based because the original loader inserts `UNK`
at ID 0. Unseen words also map to ID 0. Since padding also uses zero, M1
must use `n_words` and `n_msgs` for masks.

M2 should reproduce the original `DataPipe` sequence: for a target date, read
calendar dates target-5 through target-1; collect records in ascending date
order; assign a date to the **first trading date strictly after it**;
concatenate messages in date/file order; clip each trading step to 30; pad
word IDs to `[T,30,40]` with zero and carry `n_words`, `n_msgs`, `ss_index`.
M3 already clips each date to 30 nonempty messages and each message to 40
words. If any aligned trading step has zero messages, drop the sample.
Preserve stock and target date for evaluation.

For a real AAPL target of 2014-01-07, the source loader's step dates are
2014-01-03, 2014-01-06, 2014-01-07. Their price feature rows are 2014-01-02,
2014-01-03, 2014-01-06. The first 2014-01-02 tweet maps to the 2014-01-03
step and is at `[0,0,:18]` in the unbatched word tensor. The aligned message
counts are `[30,30,23]`. The target's movement is -0.007151, class 0 (fall).

## Evaluation handoff from M4

`evaluate.py` accepts a UTF-8 CSV with header:

```text
split,stock,target_date,y_true,y_pred
```

One row is one eligible target sample, with `split` `dev` or `test`, stock
symbol, ISO date, and class IDs 0 (fall) or 1 (rise). `y_pred` is the argmax
of M4's two-class target prediction, and `y_true` is the target label from
M2. Extra columns such as probabilities and checkpoint ID are allowed. The
evaluator checks unique stock/date keys, split dates, and labels against the
original price movement file and dead band. Keep a separate file per split.
Select the checkpoint using dev only, then report test for that checkpoint:

```powershell
python actual_project/m3_workspace/evaluate.py PATH_TO_DEV.csv --split dev
python actual_project/m3_workspace/evaluate.py PATH_TO_TEST.csv --split test
```

Confusion matrix rows are true `[fall,rise]`; columns are predicted
`[fall,rise]`, so it is `[[TN,FP],[FN,TP]]`. MCC and precision/recall/F1 are
`null` when their denominators are zero. No E0 predictions or checkpoint exist
in this checkout yet.

## Source differences requiring M2 attention

The master plan PDF page 7 says four previous **trading** days plus target,
fixed `T=5`, and maps tweet dates with `t >= d`. The original loader instead
uses `target - 4 calendar days` for price step dates (`DataPipe.py:154,169`),
sets `T=len(ts)` (`:179`), and maps each tweet date with `d < ts[t]`
(`:270-275`). This changes weekend/holiday samples and whether a same-day
tweet can enter its own step. The plan's batch table also calls zero only
`PAD`, while `DataPipe.py:79-91,46-49,226-228` uses zero for both `UNK`
and padding. M2 should match the source for E0 and document any deliberate
five-trading-step change as a deviation.

The plan's page 7 `stock_idx` range `[0,86]` assumes 87 stocks. The original
`config.yml:63-72` lists 88 price stocks, including GMRE, while 87 have
tweet directories. `DataPipe.py:375,443-445` still builds stock IDs from
the full list, so M2/M4 should retain that stock-ID mapping even if GMRE
yields no eligible samples.

The team flowchart PDF page 6 requires selecting checkpoints on validation.
The original `Executor.py:175-181` saves periodically and then measures dev;
`Executor.py:189-199` restores the latest checkpoint for test. It does not
choose the best dev checkpoint. M4 owns adding explicit dev-based selection
for the team's E0 evaluation protocol and reporting the selected checkpoint.

The preprocessed tokenization is already done. The original raw tweet
cleaning/tokenization script is not present, so this reader starts from
preprocessed records, as `DataPipe.py:230-238` does.

## Reproduce independent checks

From the parent `Project` directory:

```powershell
python actual_project/m3_workspace/tweet.py AAPL 2014-01-02
python actual_project/m3_workspace/audit_tweets.py
python actual_project/m3_workspace/check_m3.py
```

`check_m3.py` checks the real AAPL tweet and exercises evaluator arithmetic
using a previous-price-movement diagnostic predictor. Its output is only a
code check, never an E0 baseline score.

The 2026-11-22 audit found 87 tweet stock directories and 88 price stocks;
GMRE has no tweet directory. There are 25,711 tweet date files from
2014-01-01 through 2015-12-31, across 698 distinct calendar dates. By the
configured split date ranges, there are 20,118 train, 2,211 dev, and 3,382
test date files (these are **files**, not eligible model samples). There are
zero empty date files, 350 files with more than 30 usable messages, and 1,835
selected messages with more than 40 words. After per-file clipping, there
are 97,052 messages and 27,274 unknown-word IDs; IDs range 0..29,866 with
a 29,867-row embedding vocabulary. The parser was compared against the
independent full-data scan on 103 sampled files. Sparse examples are AGFS
(2 files), SPLP (8), and PICO (16), versus AAPL (696).
