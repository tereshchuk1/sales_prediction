"""Model selection: compare models, tune hyperparameters, pick the best one.

Run: uv run python -m sales_prediction.experiment [--trials 8] [--submission]

Stages
    1. all models with default parameters, 3-fold time CV (validation months 30, 31, 32)
    2. random search for every model, 2-fold time CV (months 31, 32) to save time
    3. tuned models again on the 3-fold CV -> the best model is chosen by mean RMSE
    4. every tuned model is checked once on the holdout month 33 (not used for choosing)
    5. (optional) the best model is trained on months up to 33 and predicts month 34
"""

import argparse
import json
import time
import warnings
from pathlib import Path

import pandas as pd

from sales_prediction.models import MODELS
from sales_prediction.split import BLOCK
from sales_prediction.tuning import RandomSearchTuner
from sales_prediction.validation import Validator

warnings.filterwarnings("ignore", category=FutureWarning)


class ModelSelectionExperiment:
    def __init__(
        self,
        features_path="data/features.parquet",
        report_dir="reports",
        models=tuple(MODELS),
        window=12,
        n_trials=8,
        random_state=42,
    ):
        self.features_path = features_path
        self.report_dir = Path(report_dir)
        self.model_classes = {name: MODELS[name] for name in models}
        self.n_trials = n_trials
        self.random_state = random_state
        # sliding window of 12 months: same train size in every fold,
        # keeps one full seasonal cycle and keeps training time reasonable
        self.validator = Validator(n_splits=3, window=window)
        self.tune_validator = Validator(n_splits=2, window=window)
        self.default_results = {}
        self.tuners = {}
        self.tuned_results = {}
        self.holdout_results = {}

    def load(self):
        self.df = pd.read_parquet(self.features_path)
        print(f"features: {self.df.shape}")
        print(self.validator.describe(self.df))
        return self

    def compare_defaults(self):
        print("\n=== 1. default parameters, 3-fold CV ===")
        for name, cls in self.model_classes.items():
            res = self.validator.cross_validate(
                cls(random_state=self.random_state), self.df
            )
            self.default_results[name] = res
            print(res.summary(), f"({res.fit_seconds:.0f}s)", flush=True)
        return self

    def tune(self):
        print("\n=== 2. random search, 2-fold CV ===")
        for name, cls in self.model_classes.items():
            if not cls.param_space:
                continue
            tuner = RandomSearchTuner(
                cls, self.tune_validator, self.n_trials, self.random_state
            )
            self.tuners[name] = tuner.fit(self.df)
            print(
                f"  best {name}: {tuner.best.mean:.4f} {tuner.best_params}", flush=True
            )
        return self

    def best_model(self, name):
        if name in self.tuners:
            return self.tuners[name].best_model()
        return self.model_classes[name](random_state=self.random_state)

    def compare_tuned(self):
        print("\n=== 3. tuned models, 3-fold CV ===")
        for name in self.model_classes:
            model = self.best_model(name)
            if model.get_params() == self.default_results[name].params:
                res = self.default_results[name]  # tuning kept the defaults
            else:
                res = self.validator.cross_validate(model, self.df)
            self.tuned_results[name] = res
            print(res.summary(), flush=True)
        self.best_name = min(
            self.tuned_results, key=lambda n: self.tuned_results[n].mean
        )
        print(f"best model by CV: {self.best_name}")
        return self

    def check_holdout(self):
        print("\n=== 4. holdout month 33 ===")
        for name in self.model_classes:
            res = self.validator.evaluate_holdout(self.best_model(name), self.df)
            self.holdout_results[name] = res
            print(res.summary(), flush=True)
        return self

    def make_submission(self, ids_path="data/test_ids.parquet"):
        print("\n=== 5. submission ===")
        test_block = self.validator.holdout_month + 1
        train = self.df[self.df[BLOCK] <= self.validator.holdout_month]
        if self.validator.window:
            train = train[
                train[BLOCK] > self.validator.holdout_month - self.validator.window
            ]
        test = self.df[self.df[BLOCK] == test_block]
        model = self.best_model(self.best_name).fit(train)
        pred = test[["shop_id", "item_id"]].assign(item_cnt_month=model.predict(test))
        ids = pd.read_parquet(ids_path)
        sub = ids.merge(pred, on=["shop_id", "item_id"], how="left")
        # all test pairs are in the grid, but be safe
        sub["item_cnt_month"] = sub["item_cnt_month"].fillna(0)
        path = self.report_dir / "submission.csv"
        sub[["ID", "item_cnt_month"]].to_csv(path, index=False)
        print(f"saved {path} ({len(sub)} rows)")
        return self

    def results_table(self):
        rows = []
        for name in self.model_classes:
            d, t, h = (
                self.default_results[name],
                self.tuned_results[name],
                self.holdout_results.get(name),
            )
            row = {
                "model": name,
                "cv_default": d.mean,
                "cv_tuned": t.mean,
                "cv_tuned_std": t.std,
                **{f"fold_{f.val_month}": f.score for f in t.folds},
                "holdout_33": h.mean if h else None,
                "fit_seconds_3_folds": t.fit_seconds,
            }
            rows.append(row)
        return pd.DataFrame(rows).sort_values("cv_tuned").reset_index(drop=True)

    def save(self):
        self.report_dir.mkdir(exist_ok=True)
        table = self.results_table()
        table.to_csv(self.report_dir / "model_comparison.csv", index=False)
        for name, tuner in self.tuners.items():
            tuner.history().to_csv(self.report_dir / f"tuning_{name}.csv", index=False)
        best_params = {n: self.best_model(n).get_params() for n in self.model_classes}
        (self.report_dir / "best_params.json").write_text(
            json.dumps(best_params, indent=2)
        )
        print("\n", table.round(4).to_string(index=False))
        return self

    def run(self, submission=False):
        start = time.perf_counter()
        self.load().compare_defaults().tune().compare_tuned().check_holdout().save()
        if submission:
            self.make_submission()
        print(f"\ndone in {(time.perf_counter() - start) / 60:.1f} min")
        return self


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--trials", type=int, default=8, help="random search trials per model"
    )
    parser.add_argument("--window", type=int, default=12, help="train months per fold")
    parser.add_argument("--models", nargs="+", default=list(MODELS))
    parser.add_argument("--submission", action="store_true")
    args = parser.parse_args()
    ModelSelectionExperiment(
        models=args.models, window=args.window, n_trials=args.trials
    ).run(submission=args.submission)


if __name__ == "__main__":
    main()
