"""Reproducible checks on real tweet and price records (not E0 results)."""
from evaluate import evaluate, price_movements
from tweet import TweetReader


reader = TweetReader()
record = reader.get("AAPL", "2014-01-02")
first = record["messages"][0]
assert record["n_msgs"] == 30
assert first == {
    "word_ids": [10377, 7341, 13080, 22140, 3353, 322, 10817, 1714,
                 11044, 2435, 13776, 14182, 22974, 17567, 8197, 27605,
                 28881, 25473],
    "n_words": 18, "ss_index": 1, "source_line": 1,
}
assert reader.get("GMRE", "2014-01-02")["n_msgs"] == 0

# A previous-day movement sign is only a real-data diagnostic prediction.
# It exercises all metric branches without pretending a trained model exists.
movements = price_movements("AAPL")
dates = sorted(movements)
rows = []
for previous, current in zip(dates, dates[1:]):
    movement = movements[current]
    if "2015-10-01" <= current < "2016-01-01" and not -0.005 <= movement < 0.0055:
        rows.append({"split": "test", "stock": "AAPL", "target_date": current,
                     "y_true": str(int(movement > 1e-7)),
                     "y_pred": str(int(movements[previous] > 1e-7))})
result = evaluate(rows, "test")
matrix = result["confusion_matrix"]
assert all(cell > 0 for line in matrix for cell in line)
assert result["n"] == len(rows)
assert result["accuracy"] == (matrix[0][0] + matrix[1][1]) / len(rows)
assert result["precision"] == matrix[1][1] / (matrix[0][1] + matrix[1][1])
assert result["recall"] == matrix[1][1] / (matrix[1][0] + matrix[1][1])
print("M3 parser and metric checks passed on real source records; no E0 model scored.")
