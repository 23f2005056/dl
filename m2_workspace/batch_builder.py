
"""Create StockNet batches from individual stock samples."""

import random
import numpy as np


class StockNetBatchBuilder:
    """
    Reproduce original DataPipe batching behaviour
    while retaining evaluation metadata.
    """

    def __init__(
        self,
        sample_builder,
        stock_symbols,
        batch_size=32,
        max_n_days=5,
        max_n_msgs=30,
        max_n_words=40,
        y_size=2,
    ):
        self.sample_builder = sample_builder
        self.stock_symbols = list(stock_symbols)

        self.batch_size = batch_size
        self.max_n_days = max_n_days
        self.max_n_msgs = max_n_msgs
        self.max_n_words = max_n_words
        self.y_size = y_size

        # Map numerical stock IDs back to stock symbols.
        self.stock_id_to_symbol = {
            stock_id: symbol
            for symbol, stock_id
            in sample_builder.mappings.stock_id_dict.items()
        }

    def get_batch_size(self, phase):
        """Use original DataPipe phase-specific batch sizes."""

        if phase == "train":
            return self.batch_size
        elif phase == "unit_test":
            return 5
        else:
            return 1

    def iter_batches(self, phase):
        """
        Yield batches containing randomly selected stock samples.
        """

        batch_size = self.get_batch_size(phase)

        generators = [
            iter(self.sample_builder.iter_samples(stock, phase))
            for stock in self.stock_symbols
        ]

        while generators:

            stock_batch = np.zeros(
                (batch_size,), dtype=np.int32
            )

            T_batch = np.zeros(
                (batch_size,), dtype=np.int32
            )

            y_batch = np.zeros(
                (batch_size, self.max_n_days, self.y_size),
                dtype=np.float32,
            )

            main_mv_percent_batch = np.zeros(
                (batch_size,), dtype=np.float32
            )

            mv_percent_batch = np.zeros(
                (batch_size, self.max_n_days),
                dtype=np.float32
            )

            price_batch = np.zeros(
                (batch_size, self.max_n_days, 3),
                dtype=np.float32
            )

            word_batch = np.zeros(
                (
                    batch_size,
                    self.max_n_days,
                    self.max_n_msgs,
                    self.max_n_words,
                ),
                dtype=np.int32,
            )

            ss_index_batch = np.zeros(
                (
                    batch_size,
                    self.max_n_days,
                    self.max_n_msgs,
                ),
                dtype=np.int32,
            )

            n_msgs_batch = np.zeros(
                (batch_size, self.max_n_days),
                dtype=np.int32
            )

            n_words_batch = np.zeros(
                (
                    batch_size,
                    self.max_n_days,
                    self.max_n_msgs,
                ),
                dtype=np.int32,
            )

            # New evaluation metadata.
            stock_names = []
            target_dates = []

            targets = np.zeros(
                (batch_size,), dtype=np.int32
            )

            sample_id = 0

            while sample_id < batch_size and generators:

                gen_id = random.randint(
                    0, len(generators) - 1
                )

                try:
                    sample = next(generators[gen_id])

                    T = sample["T"]
                    stock_id = sample["stock"]

                    stock_batch[sample_id] = stock_id
                    T_batch[sample_id] = T

                    y_batch[sample_id, :T] = sample["ys"]

                    main_mv_percent_batch[sample_id] = (
                        sample["main_mv_percent"]
                    )

                    mv_percent_batch[sample_id, :T] = (
                        sample["mv_percents"]
                    )

                    price_batch[sample_id, :T] = (
                        sample["prices"]
                    )

                    word_batch[sample_id, :T] = (
                        sample["msgs"]
                    )

                    ss_index_batch[sample_id, :T] = (
                        sample["ss_indices"]
                    )

                    n_msgs_batch[sample_id, :T] = (
                        sample["n_msgs"]
                    )

                    n_words_batch[sample_id, :T] = (
                        sample["n_words"]
                    )

                    # Evaluation metadata.
                    stock_names.append(
                        self.stock_id_to_symbol[stock_id]
                    )

                    date = sample["main_target_date"]

                    target_dates.append(
                        date.isoformat()
                        if hasattr(date, "isoformat")
                        else str(date)
                    )

                    movement = float(
                        sample["main_mv_percent"]
                    )

                    targets[sample_id] = int(
                        movement > 1e-7
                    )

                    sample_id += 1

                except StopIteration:
                    del generators[gen_id]

            # Preserve original behaviour:
            # discard incomplete final batches.
            if sample_id < batch_size:
                return

            yield {
                "batch_size": sample_id,

                # Original model inputs.
                "stock_batch": stock_batch,
                "T_batch": T_batch,
                "y_batch": y_batch,
                "main_mv_percent_batch": main_mv_percent_batch,
                "mv_percent_batch": mv_percent_batch,
                "price_batch": price_batch,
                "word_batch": word_batch,
                "ss_index_batch": ss_index_batch,
                "n_msgs_batch": n_msgs_batch,
                "n_words_batch": n_words_batch,

                # New evaluation metadata.
                "stock_names": stock_names,
                "target_dates": target_dates,
                "targets": targets,
            }
