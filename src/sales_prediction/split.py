"""Time-based splits and preprocessing fitted on the train part only."""

import numpy as np
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

BLOCK = "date_block_num"
TARGET = "item_cnt_month"
CAT_COLS = ["city", "channel", "category_group"]
# identifiers: numbers without order, the model would just memorise them
ID_COLS = ["shop_id", "item_id"]


class ChronologicalSplitter:
    def __init__(self, val_block=33, test_block=34):
        self.val_block = val_block
        self.test_block = test_block

    def split(self, df):
        train = df[df[BLOCK] < self.val_block]
        val = df[df[BLOCK] == self.val_block]
        test = df[df[BLOCK] == self.test_block]
        return train, val, test


class MonthTimeSeriesSplit:
    """TimeSeriesSplit over months (a month is never cut in two).

    With n_splits=3 and last month 32:
        fold 1: train <= 29, validation 30
        fold 2: train <= 30, validation 31
        fold 3: train <= 31, validation 32
    """

    def __init__(self, n_splits=3, window=None, gap=0):
        self.n_splits = n_splits
        self.window = window
        self.gap = gap

    def split(self, df):
        df = df[df[TARGET].notna()]  # only months with a target
        for train_m, val_m in self.month_splits(df[BLOCK].unique()):
            train_idx = df.index[df[BLOCK].isin(train_m)]
            val_idx = df.index[df[BLOCK].isin(val_m)]
            yield train_idx, val_idx

    def month_splits(self, months):
        # same as sklearn TimeSeriesSplit(test_size=1), but n_splits=1 is allowed
        months = np.sort(months)
        for k in range(self.n_splits, 0, -1):
            val_pos = len(months) - k
            train_end = val_pos - self.gap
            train_start = max(0, train_end - self.window) if self.window else 0
            if train_end <= train_start:
                raise ValueError("not enough months for this number of splits")
            yield months[train_start:train_end], months[val_pos : val_pos + 1]


class Preprocessor:
    """fit() only on the train part, transform() on train / validation / test.

    - text categories -> numbers (categories not seen in train get -1),
      or one-hot columns if one_hot=True (better for linear models)
    - NaN -> median of train (only if fill_na=True, trees can work with NaN)
    - standard scaling (only if scale=True, needed for linear models and SVR, not for trees)
    - ID columns are dropped
    """

    def __init__(self, fill_na=False, scale=False, one_hot=False, drop_cols=ID_COLS):
        self.fill_na = fill_na
        self.scale = scale
        self.one_hot = one_hot
        self.drop_cols = list(drop_cols)

    def fit(self, train):
        skip = [BLOCK, TARGET] + CAT_COLS + self.drop_cols
        self.num_cols = [c for c in train.columns if c not in skip]
        if self.one_hot:
            self.encoder = OneHotEncoder(
                handle_unknown="ignore", sparse_output=False, dtype=np.float32
            )
        else:
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

    @property
    def feature_names(self):
        return self.num_cols + list(self.encoder.get_feature_names_out(CAT_COLS))

    def transform(self, df):
        X = df[self.num_cols].astype("float32")
        if self.fill_na or self.scale:
            X = X.fillna(self.medians)
        if self.scale:
            X[self.num_cols] = self.scaler.transform(X)
        cats = self.encoder.transform(df[CAT_COLS]).astype("float32")
        X = np.hstack([X.to_numpy("float32"), cats])
        return X
