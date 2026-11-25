"""Models with one interface: fit(train_df) -> self, predict(df) -> array.

Every model owns its Preprocessor, so preprocessing is always fitted on the train
part of the current fold only (no leakage from validation into medians / scaler).
"""

from abc import ABC, abstractmethod
from typing import ClassVar

import numpy as np
from scipy.stats import loguniform, randint, uniform
from sklearn.ensemble import RandomForestRegressor
from sklearn.kernel_approximation import Nystroem
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.svm import LinearSVR
from xgboost import XGBRegressor

from sales_prediction.split import TARGET, Preprocessor
from sales_prediction.validation import CLIP


class SalesModel(ABC):
    """Base class: sampling of train rows, preprocessing, clipping of predictions.

    Subclasses set:
        name - short name for reports
        default_params - hyperparameters used without tuning
        param_space - distributions for random search
        preprocessing - kwargs for Preprocessor
        max_train_rows - train rows are randomly sampled down to this (None = all)
    """

    name = "model"
    default_params: ClassVar[dict] = {}
    param_space: ClassVar[dict] = {}
    preprocessing: ClassVar[dict] = {}
    max_train_rows = None

    def __init__(self, random_state=42, **params):
        unknown = set(params) - set(self.default_params)
        if unknown:
            raise ValueError(f"{self.name}: unknown params {unknown}")
        self.params = {**self.default_params, **params}
        self.random_state = random_state

    def get_params(self):
        return dict(self.params)

    def _sample(self, train):
        if self.max_train_rows and len(train) > self.max_train_rows:
            return train.sample(self.max_train_rows, random_state=self.random_state)
        return train

    def fit(self, train):
        train = self._sample(train)
        self.preprocessor = Preprocessor(**self.preprocessing).fit(train)
        X = self.preprocessor.transform(train)
        self.estimator = self.build()
        self.estimator.fit(X, train[TARGET].to_numpy("float32"))
        return self

    def predict(self, df):
        X = self.preprocessor.transform(df)
        return np.clip(self.estimator.predict(X), *CLIP)

    @abstractmethod
    def build(self):
        """Returns a new unfitted sklearn-like estimator."""

    def __repr__(self):
        return f"{type(self).__name__}({self.params})"


class LastMonthBaseline(SalesModel):
    """Prediction = sales of the pair in the previous month (0 if none)."""

    name = "baseline_lag1"

    def fit(self, train):
        return self

    def predict(self, df):
        return np.clip(df["cnt_lag_1"].fillna(0).to_numpy(), *CLIP)

    def build(self):
        return None


class LinearRegressionModel(SalesModel):
    """Linear regression on scaled features with one-hot categories."""

    name = "linear_regression"
    default_params: ClassVar[dict] = {"alpha": 0.0}
    param_space: ClassVar[dict] = {"alpha": [0.0, 0.1, 1.0, 10.0, 100.0, 1000.0]}
    preprocessing: ClassVar[dict] = {"fill_na": True, "scale": True, "one_hot": True}
    max_train_rows = 2_000_000

    def build(self):
        if self.params["alpha"] == 0:
            return LinearRegression()
        return Ridge(alpha=self.params["alpha"])


class SVRModel(SalesModel):
    """Support vector regression with an RBF kernel"""

    name = "svr"
    default_params: ClassVar[dict] = {
        "C": 1.0,
        "epsilon": 0.1,
        "gamma": 0.01,
        "n_components": 300,
    }
    param_space: ClassVar[dict] = {
        "C": loguniform(0.01, 10),
        "epsilon": [0.0, 0.05, 0.1, 0.3],
        "gamma": loguniform(1e-3, 1e-1),
        "n_components": [200, 300, 500],
    }
    preprocessing: ClassVar[dict] = {"fill_na": True, "scale": True, "one_hot": True}
    max_train_rows = 200_000

    def build(self):
        p = self.params
        return make_pipeline(
            Nystroem(
                kernel="rbf",
                gamma=p["gamma"],
                n_components=p["n_components"],
                random_state=self.random_state,
            ),
            LinearSVR(
                C=p["C"],
                epsilon=p["epsilon"],
                loss="epsilon_insensitive",
                dual=True,
                max_iter=5000,
                random_state=self.random_state,
            ),
        )


class RandomForestModel(SalesModel):
    """Random forest"""

    name = "random_forest"
    default_params: ClassVar[dict] = {
        "n_estimators": 100,
        "max_depth": 14,
        "min_samples_leaf": 20,
        "max_features": 0.5,
        "max_samples": 0.2,
    }
    param_space: ClassVar[dict] = {
        "n_estimators": randint(50, 200),
        "max_depth": [8, 10, 12, 14, 16, 20],
        "min_samples_leaf": [5, 10, 20, 50, 100],
        "max_features": [0.3, 0.5, 0.7, "sqrt"],
        "max_samples": [0.1, 0.2, 0.3],
    }
    preprocessing: ClassVar[dict] = {"fill_na": True}
    max_train_rows = 1_000_000

    def build(self):
        return RandomForestRegressor(
            **self.params, n_jobs=-1, random_state=self.random_state
        )


class XGBoostModel(SalesModel):
    """Gradient boosting. Missing values are handled natively."""

    name = "xgboost"
    default_params: ClassVar[dict] = {
        "n_estimators": 300,
        "learning_rate": 0.1,
        "max_depth": 8,
        "min_child_weight": 50,
        "subsample": 0.8,
        "colsample_bytree": 0.7,
        "reg_lambda": 1.0,
    }
    param_space: ClassVar[dict] = {
        "n_estimators": randint(200, 600),
        "learning_rate": loguniform(0.02, 0.2),
        "max_depth": randint(5, 11),
        "min_child_weight": [1, 10, 50, 100, 300],
        "subsample": uniform(0.6, 0.4),
        "colsample_bytree": uniform(0.4, 0.6),
        "reg_lambda": loguniform(0.1, 10),
    }
    preprocessing: ClassVar[dict] = {}
    max_train_rows = None

    def build(self):
        return XGBRegressor(
            **self.params,
            tree_method="hist",
            n_jobs=-1,
            random_state=self.random_state,
        )

    def feature_importance(self):
        names = self.preprocessor.feature_names
        return dict(
            sorted(
                zip(names, self.estimator.feature_importances_),
                key=lambda kv: -kv[1],
            )
        )


MODELS = {
    cls.name: cls
    for cls in [
        LastMonthBaseline,
        LinearRegressionModel,
        SVRModel,
        RandomForestModel,
        XGBoostModel,
    ]
}
