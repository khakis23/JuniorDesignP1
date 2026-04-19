import random
from abc import ABC, abstractmethod
from typing import Callable

# sklearn
from sklearn.linear_model import *
from sklearn.model_selection import train_test_split, TimeSeriesSplit, cross_val_score
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import GradientBoostingRegressor

# other
import pandas as pd
import numpy as np
from scipy.stats import bootstrap
import matplotlib.pyplot as plt


class IModel(ABC):

    def __init__(self, features: list[str], targets: list[str], data: pd.DataFrame, data_test: pd.DataFrame=None, plot_func: Callable=None):
        self.model = None
        self.features = features
        self.targets = targets
        self._plot_func = plot_func

        self._pre_split = data_test is not None

        if self._pre_split:
            # If test data is provided, map train to data and test to data_test
            self._y = {
                "full": pd.concat([data[targets], data_test[targets]]),
                "train": data[targets],
                "test": data_test[targets],
            }
            self._x = {
                "full": pd.concat([data[features], data_test[features]]),
                "train": data[features],
                "test": data_test[features],
            }
        else:
            # Default behavior for single dataset
            self._y = {
                "full": data[targets],
                "test": None,
                "train": None,
            }
            self._x = {
                "full": data[features],
                "test": None,
                "train": None,
            }

        self._scores: dict[str, float] = {}
        self._parameters: dict[str, any] = {}
        self._test_size: float = 0.0
        self._random_state: int | None = None

        self._predictions: np.ndarray | None = None

    @abstractmethod
    def train_and_fit(self, tts: float = 0.0, random_state=None, **kwargs):
        """
        IMPLEMENTATION:
            - Call super() on this method.
            - Implement this method to: use either train_test_split() or train final model depending on if tts > 0.
            - Call self._fit() after training model.

        :param random_state:
        :param tts:     Train test split ratio
        :param kwargs:  Parameters to train the model  (e.g.  {random_state=42, hidden_layer_sizes=(100, 100), ...} )
        """
        rand_or_none = {}
        if random_state:
            self._random_state = random_state
            rand_or_none = {"random_state": random_state}

        # Bypass TTS entirely if data_test was provided in constructor
        if self._pre_split:
            self._test_size = -1.0  # Indicator that it is pre-split
        elif tts > 0:
            self._test_size = tts
            self._x["train"], self._x["test"], self._y["train"], self._y["test"] = train_test_split(
                self._x["full"], self._y["full"], test_size=tts, **rand_or_none,
            )
        else:
            self._test_size = 0.0

    def _fit(self):
        """
        IMPLEMENTATION: call this method after training model in train_and_fit()
        """
        # Fit on train if TTS occurred OR if data was pre-split
        if self._test_size > 0 or self._pre_split:
            self.model.fit(self._x["train"], self._y["train"])
        else:
            self.model.fit(self._x["full"], self._y["full"])

    def _score(self, folds: int = 5):
        cv_r2 = np.nan

        try:  # TODO broken with non-sklearn models!
            ts_cv = TimeSeriesSplit(n_splits=folds)
            cv_scores = cross_val_score(
                self.model,
                self._x["train"],
                self._y["train"].values.ravel() if len(self.targets) == 1 else self._y["train"].values,
                scoring="r2",
                cv=ts_cv,
            )
            cv_r2 = np.mean(cv_scores)
        except Exception:
            cv_r2 = np.nan

        # Added multioutput handling for multi-target datasets
        self._scores = {
            "R2": r2_score(self._y["test"], self._predictions, multioutput="uniform_average"),
            "RMSE": mean_squared_error(y_true=self._y["test"], y_pred=self._predictions,
                                       multioutput="uniform_average") ** 0.5,
            "MAE": mean_absolute_error(y_true=self._y["test"], y_pred=self._predictions, multioutput="uniform_average"),
            "CI": self._get_bootstrap(r2_score)
        }
        if cv_r2:
            self._scores["CV R2"] = cv_r2

    def _get_bootstrap(self, score_func: Callable, n_resamples: int = 1000) -> tuple[float, float]:
        y_true = np.ravel(self._y["test"].values)
        y_pred = np.ravel(self._predictions)

        # bootstrap requires at least 2 samples
        if len(y_true) < 2:
            return np.nan, np.nan

        # Ravel ensures multidimensional outputs are flattened properly for scipy's bootstrap pairing
        res = bootstrap(
            (y_true, y_pred),
            score_func,
            vectorized=False,
            paired=True,
            n_resamples=n_resamples,
            method='percentile',
            random_state=self._random_state)
        return res.confidence_interval.low, res.confidence_interval.high

    def predict(self, x: pd.DataFrame = None) -> np.ndarray:
        """
        Predict using parameter X or full dataset (if final model), or X_test (if testing model).

        :param x:  (optional) X to predict with
        :return:    predictions
        """
        if self.model is None:
            raise ValueError("Model has not been trained!")

        # Evaluate if it is a testing model (TTS > 0 or Pre-split)
        if self._test_size > 0 or self._pre_split:
            self._predictions = self.model.predict(self._x["test"])
            self._score()
        # final models
        else:
            x = self._x["full"] if x is None else x
            self._predictions = self.model.predict(x)

        return self._predictions

    def print_scores(self) -> None:
        longest_name = max((len(sn) for sn in self._scores.keys()), default=0)
        longest_score = 0

        for sv in self._scores.values():
            if isinstance(sv, (list, np.ndarray, tuple)):
                val_str = ", ".join(str(round(x, 3)) for x in sv)
            else:
                val_str = str(round(sv, 3))
            longest_score = max(longest_score, len(val_str))

        for sn, sv in self._scores.items():
            if isinstance(sv, (list, np.ndarray, tuple)):
                val_str = ", ".join(str(round(x, 3)) for x in sv)
            else:
                val_str = str(round(sv, 3))
            print(f"{sn:<{longest_name + 1}} | {val_str:>{longest_score + 1}}")

    def print_parameters(self) -> None:
        if not self._parameters:
            return

        longest_name = max((len(pn) for pn in self._parameters.keys()), default=0)
        longest_val = 0

        # TODO separate these duplicates into a helper function!
        for pv in self._parameters.values():
            if isinstance(pv, (list, np.ndarray, tuple)):
                # only attempt to round if the element is actually a number
                val_str = ", ".join(
                    str(round(x, 4))
                        if isinstance(x, (int, float, np.number))
                        else str(x)
                            for x in pv)
            else:
                val_str = str(round(pv, 4)) if isinstance(pv, (int, float)) else str(pv)
            longest_val = max(longest_val, len(val_str))

        for pn, pv in self._parameters.items():
            if isinstance(pv, (list, np.ndarray, tuple)):
                # only attempt to round if the element is actually a number
                val_str = ", ".join(
                    str(round(x, 4))
                    if isinstance(x, (int, float, np.number))
                    else str(x)
                    for x in pv)
            else:
                val_str = str(round(pv, 4)) if isinstance(pv, (int, float)) else str(pv)
            print(f"{pn:<{longest_name + 1}} | {val_str}")

    def get_scores(self) -> dict[str, float]:
        return self._scores

    def get_parameters(self) -> dict[str, any]:
        return self._parameters

    def get_features(self) -> list[str]:
        return self.features

    def get_targets(self) -> list[str]:
        return self.targets

    def set_plot(self, func: Callable):
        self._plot_func = func

    def plot(self, **kwargs):
        if not self._plot_func:
            print("No plot function set. Call set_plot() first.")
            return

        self._plot_func(self, self._x, self._y, **kwargs)
