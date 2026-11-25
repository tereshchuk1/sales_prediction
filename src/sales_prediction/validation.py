"""Validation schema.

    months 12 .. 32 -> cross-validation (model selection, hyperparameter tuning)
    month 33 -> used once at the end to confirm the chosen model
    month 34 -> test (no target, Kaggle submission)

Every fold has the same shape as production: train on past months, predict the next one.
"""

import time
from dataclasses import dataclass, field

import numpy as np

from sales_prediction.split import BLOCK, TARGET, MonthTimeSeriesSplit

CLIP = (0, 20)  # target is clipped like in the competition


def rmse(y_true, y_pred):
    """RMSE on clipped predictions."""
    y_pred = np.clip(y_pred, *CLIP)
    return float(np.sqrt(np.mean((np.asarray(y_true) - y_pred) ** 2)))


@dataclass
class FoldResult:
    val_month: int
    score: float
    fit_seconds: float


@dataclass
class ValidationResult:
    model: str
    params: dict
    folds: list = field(default_factory=list)

    @property
    def mean(self):
        # mean score
        return float(np.mean([f.score for f in self.folds]))

    @property
    def std(self):
        # std between months
        return float(np.std([f.score for f in self.folds]))

    @property
    def fit_seconds(self):
        # number of seconds for education
        return sum(f.fit_seconds for f in self.folds)

    def summary(self):
        folds = ", ".join(f"{f.val_month}: {f.score:.4f}" for f in self.folds)
        return f"{self.model:<18} rmse {self.mean:.4f} +- {self.std:.4f}  [{folds}]"


class Validator:
    """Time-based validation of models with fit(train_df) / predict(df).

    n_splits - number of CV folds, each fold validates on one month
    window - number of train months (sliding window), None = all past months
    holdout_month - month that CV never sees
    """

    def __init__(self, n_splits=3, window=None, holdout_month=33):
        self.window = window
        self.holdout_month = holdout_month
        self.cv = MonthTimeSeriesSplit(n_splits, window)

    def folds(self, df):
        # CV only sees months before the holdout
        part = df[df[BLOCK] < self.holdout_month]
        for train_idx, val_idx in self.cv.split(part):
            yield part.loc[train_idx], part.loc[val_idx]

    def holdout_split(self, df):
        train = df[df[BLOCK] < self.holdout_month]
        if self.window:
            train = train[train[BLOCK] >= self.holdout_month - self.window]
        return train, df[df[BLOCK] == self.holdout_month]

    def describe(self, df):
        """Months and row counts of every fold and of the holdout."""
        splits = [*self.folds(df), self.holdout_split(df)]
        return "\n".join(
            f"train months {tr[BLOCK].min()}-{tr[BLOCK].max()} ({len(tr):,} rows) "
            f"-> validation {va[BLOCK].iloc[0]} ({len(va):,} rows)"
            for tr, va in splits
        )

    def _fit_score(self, model, train, val):
        start = time.perf_counter()
        model.fit(train)
        score = rmse(val[TARGET], model.predict(val))
        return FoldResult(int(val[BLOCK].iloc[0]), score, time.perf_counter() - start)

    def cross_validate(self, model, df):
        result = ValidationResult(model.name, model.get_params())
        result.folds = [self._fit_score(model, tr, va) for tr, va in self.folds(df)]
        return result

    def evaluate_holdout(self, model, df):
        result = ValidationResult(model.name, model.get_params())
        result.folds = [self._fit_score(model, *self.holdout_split(df))]
        return result
