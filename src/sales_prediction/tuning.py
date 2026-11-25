"""Hyperparameter search scored by the time-based cross-validation."""

import pandas as pd
from sklearn.model_selection import ParameterSampler


class RandomSearchTuner:
    """Random search over model_cls.param_space.

    Every candidate is scored with validator.cross_validate (mean RMSE over the
    folds), so tuning never sees the holdout month.
    The default parameters are always tried first, so tuning can't end up worse.
    """

    def __init__(
        self, model_cls, validator, n_trials=10, random_state=42, verbose=True
    ):
        self.model_cls = model_cls
        self.validator = validator
        self.n_trials = n_trials
        self.random_state = random_state
        self.verbose = verbose
        self.results = []

    def candidates(self):
        default = dict(self.model_cls.default_params)
        yield default
        if not self.model_cls.param_space or self.n_trials <= 1:
            return
        sampler = ParameterSampler(
            self.model_cls.param_space,
            n_iter=self.n_trials - 1,
            random_state=self.random_state,
        )
        seen = [default]
        for params in sampler:
            params = {
                k: v.item() if hasattr(v, "item") else v for k, v in params.items()
            }
            params = {**default, **params}
            if params not in seen:  # small grids repeat candidates
                seen.append(params)
                yield params

    def fit(self, df):
        self.results = []
        for i, params in enumerate(self.candidates(), 1):
            model = self.model_cls(random_state=self.random_state, **params)
            res = self.validator.cross_validate(model, df)
            self.results.append(res)
            if self.verbose:
                print(
                    f"[{self.model_cls.name} {i}/{self.n_trials}]"
                    f"rmse {res.mean:.4f} +- {res.std:.4f} ({res.fit_seconds:.0f}s) {params}"
                )
        return self

    @property
    def best(self):
        return min(self.results, key=lambda r: r.mean)

    @property
    def best_params(self):
        return self.best.params

    def best_model(self):
        return self.model_cls(random_state=self.random_state, **self.best_params)

    def history(self):
        rows = [
            {"model": r.model, "rmse_mean": r.mean, "rmse_std": r.std, **r.params}
            for r in self.results
        ]
        return pd.DataFrame(rows).sort_values("rmse_mean")
