
"""Reproduce StockNet's original price and tweet alignment."""

from datetime import timedelta


def build_price_timeline(price_records, target_date, max_n_days=5):
    """Reproduce the original _get_prices_and_ts logic."""

    d_t_min = target_date - timedelta(days=max_n_days - 1)
    all_dates = sorted(price_records)

    if target_date not in price_records:
        return None

    movement = price_records[target_date]["movement"]

    if -0.005 <= movement < 0.0055:
        return None

    ts = [
        d for d in all_dates
        if d_t_min <= d <= target_date
    ]

    if not ts or ts[-1] != target_date:
        return None

    earlier_dates = [d for d in all_dates if d < d_t_min]

    if not earlier_dates:
        return None

    # One older row, followed by the historical rows.
    price_dates = [earlier_dates[-1]] + [
        d for d in ts if d < target_date
    ]

    if len(price_dates) != len(ts):
        return None

    prices = [
        [
            price_records[d]["high"],
            price_records[d]["low"],
            price_records[d]["close"],
        ]
        for d in price_dates
    ]

    labels = [
    [1.0, 0.0] 
    if price_records[d]["movement"] <= 1e-7
    else [0.0, 1.0]
    for d in ts
	]

	# Original StockNet stores historical movement class IDs here,
	# not raw movement percentages.
    mv_percents = [
  	  0 if price_records[d]["movement"] <= 1e-7 else 1
	    for d in price_dates
    ]
    return {
        "T": len(ts),
        "ts": ts,
        "price_dates": price_dates,
        "prices": prices,
        "ys": labels,
        "main_mv_percent": movement,
        "mv_percents": mv_percents,
        "target_movement": movement,
        "target_label": int(movement > 1e-7),
    }


def align_tweets_for_stock(
    stock,
    ts,
    target_date,
    tweet_reader,
    max_n_days=5,
    max_n_msgs=30,
    max_n_words=40,
):
    """Align calendar-date tweets to trading-date steps."""

    T = len(ts)

    aligned_msgs = [[] for _ in range(T)]
    aligned_ss = [[] for _ in range(T)]
    aligned_n_words = [[] for _ in range(T)]

    # Ascending calendar-date order: target-5 through target-1.
    tweet_dates = [
        target_date - timedelta(days=offset)
        for offset in range(max_n_days, 0, -1)
    ]

    for tweet_date in tweet_dates:
        record = tweet_reader.get(stock, tweet_date.isoformat())

        # First trading date strictly after the tweet date.
        step_index = next(
            (
                i for i, trading_date in enumerate(ts)
                if tweet_date < trading_date
            ),
            None,
        )

        if step_index is None:
            continue

        for message in record["messages"]:
            aligned_msgs[step_index].append(message["word_ids"])
            aligned_ss[step_index].append(message["ss_index"])
            aligned_n_words[step_index].append(message["n_words"])

    # Original source uses max_threshold = 0.
    # Every trading step must have at least one message.
    if any(len(messages) == 0 for messages in aligned_msgs):
        return None

    msgs_tensor = []
    ss_matrix = []
    n_words_matrix = []
    n_msgs_vector = []

    for step in range(T):
        messages = aligned_msgs[step][:max_n_msgs]
        ss_values = aligned_ss[step][:max_n_msgs]
        word_counts = aligned_n_words[step][:max_n_msgs]

        step_msgs = []
        step_ss = []
        step_n_words = []

        for message, ss_index, n_words in zip(
            messages, ss_values, word_counts
        ):
            padded = list(message[:max_n_words])
            padded += [0] * (max_n_words - len(padded))

            step_msgs.append(padded)
            step_ss.append(ss_index)
            step_n_words.append(n_words)

        while len(step_msgs) < max_n_msgs:
            step_msgs.append([0] * max_n_words)
            step_ss.append(0)
            step_n_words.append(0)

        msgs_tensor.append(step_msgs)
        ss_matrix.append(step_ss)
        n_words_matrix.append(step_n_words)
        n_msgs_vector.append(len(messages))

    return {
        "msgs": msgs_tensor,
        "ss_indices": ss_matrix,
        "n_words": n_words_matrix,
        "n_msgs": n_msgs_vector,
    }
