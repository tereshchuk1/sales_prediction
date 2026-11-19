"""Time-based splits and preprocessing fitted on the train part only.

Random k-fold is not used: the data is not stationary (EDA 4) and it would leak the future.
"""

import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

BLOCK = "date_block_num"
TARGET = "item_cnt_month"
CAT_COLS = ["city", "channel", "category_group"]


def chronological_split(df, val_block=33, test_block=34):
    # train: months before val_block, validation: val_block, test: test_block
    train = df[df[BLOCK] < val_block]
    val = df[df[BLOCK] == val_block]
    test = df[df[BLOCK] == test_block]
    return train, val, test


def time_series_cv(df, n_splits=3, window=None, gap=0):
    """TimeSeriesSplit over months, not over rows (a month is never cut in two).

    Expanding window, or sliding if window (number of months) is set.
    With n_splits=3 and last month 33:
        fold 1: train <= 30, validation 31
        fold 2: train <= 31, validation 32
        fold 3: train <= 32, validation 33
    """
    df = df[df[TARGET].notna()]  # only months with a target
    months = np.sort(df[BLOCK].unique())
    tss = TimeSeriesSplit(
        n_splits=n_splits, test_size=1, gap=gap, max_train_size=window
    )
    for train_m, val_m in tss.split(months):
        train_idx = df.index[df[BLOCK].isin(months[train_m])]
        val_idx = df.index[df[BLOCK].isin(months[val_m])]
        yield train_idx, val_idx


class Preprocessor:
    """fit() only on the train part, transform() on train / validation / test.

    - text categories -> numbers (categories not seen in train get -1)
    - NaN -> median of train (only if fill_na=True, trees can work with NaN)
    - standard scaling (only if scale=True, needed for linear models, not for trees)
    """

    def __init__(self, fill_na=False, scale=False):
        self.fill_na = fill_na
        self.scale = scale

    def fit(self, train):
        self.num_cols = [
            c for c in train.columns if c not in [BLOCK, TARGET] + CAT_COLS
        ]
        self.encoder = OrdinalEncoder(
            handle_unknown="use_encoded_value", unknown_value=-1
        )
        self.encoder.fit(train[CAT_COLS])
        self.medians = train[self.num_cols].median()
        if self.scale:
            self.scaler = StandardScaler().fit(
                train[self.num_cols].fillna(self.medians)
            )
        return self

    def transform(self, df):
        X = df[self.num_cols].copy()
        if self.fill_na or self.scale:
            X = X.fillna(self.medians)
        if self.scale:
            X[self.num_cols] = self.scaler.transform(X)
        X[CAT_COLS] = self.encoder.transform(df[CAT_COLS])
        return X.astype("float32")
