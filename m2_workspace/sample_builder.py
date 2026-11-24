
"""Build individual StockNet samples and generate phase-specific samples."""

import random
from datetime import date

from date_alignment import (
    build_price_timeline,
    align_tweets_for_stock,
)


class StockNetSampleBuilder:
    """
    Build StockNet samples and generate samples for train/dev/test phases.
    """

    PHASE_DATES = {
        "train": ("2014-01-01", "2015-08-01"),
        "dev": ("2015-08-01", "2015-10-01"),
        "test": ("2015-10-01", "2016-01-01"),
        "whole": ("2014-01-01", "2016-01-01"),
        "unit_test": ("2012-07-23", "2012-08-05"),
    }

    def __init__(
        self,
        mappings,
        price_reader,
        tweet_reader,
        max_n_days=5,
        max_n_msgs=30,
        max_n_words=40,
        shuffle=True,
    ):
        self.mappings = mappings
        self.price_reader = price_reader
        self.tweet_reader = tweet_reader

        self.max_n_days = max_n_days
        self.max_n_msgs = max_n_msgs
        self.max_n_words = max_n_words
        self.shuffle = bool(shuffle)

    def build(self, stock, target_date, price_records=None):
        """
        Build one sample for a stock and target date.

        Returns a sample dictionary, or None if the sample
        is not eligible.
        """

        if stock not in self.mappings.stock_id_dict:
            raise ValueError(f"Unknown stock symbol: {stock}")

        stock_id = self.mappings.stock_id_dict[stock]

        # Read prices unless already supplied by iter_samples().
        if price_records is None:
            price_records = self.price_reader.read(stock)

        price_info = build_price_timeline(
            price_records=price_records,
            target_date=target_date,
            max_n_days=self.max_n_days,
        )

        if price_info is None:
            return None

        tweet_info = align_tweets_for_stock(
            stock=stock,
            ts=price_info["ts"],
            target_date=target_date,
            tweet_reader=self.tweet_reader,
            max_n_days=self.max_n_days,
            max_n_msgs=self.max_n_msgs,
            max_n_words=self.max_n_words,
        )

        # Original DataPipe skips samples with invalid tweet alignment.
        if tweet_info is None:
            return None

        return {
            "stock": stock_id,
            "main_target_date": target_date.isoformat(),
            "T": price_info["T"],
            "ys": price_info["ys"],
            "main_mv_percent": price_info["main_mv_percent"],
            "mv_percents": price_info["mv_percents"],
            "prices": price_info["prices"],
            "msgs": tweet_info["msgs"],
            "ss_indices": tweet_info["ss_indices"],
            "n_words": tweet_info["n_words"],
            "n_msgs": tweet_info["n_msgs"],
        }

    def iter_samples(self, stock, phase, shuffle=None):
        """
        Yield valid samples for one stock and one phase.

        The phase end date is exclusive, matching the original DataPipe.
        """

        if phase not in self.PHASE_DATES:
            raise ValueError(
                f"Unknown phase: {phase}. "
                f"Choose from {list(self.PHASE_DATES)}"
            )

        if stock not in self.mappings.stock_id_dict:
            raise ValueError(f"Unknown stock symbol: {stock}")

        start_date, end_date = self.PHASE_DATES[phase]

        # Read this stock's price/movement file once.
        price_records = self.price_reader.read(stock)

        # Preserve the source file's original date order.
        target_dates = [
            target_date
            for target_date in price_records.keys()
            if start_date <= target_date.isoformat() < end_date
        ]

        # Match original DataPipe's optional date shuffling.
        should_shuffle = self.shuffle if shuffle is None else bool(shuffle)

        if should_shuffle:
            random.shuffle(target_dates)

        for target_date in target_dates:
            sample = self.build(
                stock=stock,
                target_date=target_date,
                price_records=price_records,
            )

            if sample is not None:
                yield sample
