from abc import ABC, abstractmethod
from typing import Callable

from Data.ModelData.ModelData import ModelData

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


"""
Implementation Usage:
    

User Usage:
    1. Instantiate with features and data
    2. train_and_fit() (optional tts (train test split), otherwise trains on full dataset)
    3. predict()
    4. Display data
        - plot
        - print_scores()
        - get_scores()
"""
class IModel(ABC):

    def __init__(self, features: list[str], data: ModelData=ModelData()):
        self.model = None
        self.features = features
        self._elevation_df = data.weather["sunelevation"]   # for clamping

        self._y = {
            "full": data.energy,   # full dataset
            "test": None,   # optional
            "train": None,  # optional
        }
        self._x = {
            "full": data.features[features],
            "test": None,
            "train": None,
        }

        self._scores: dict[str, float] = {}
        self._parameters: dict[str, any] = {}
        self._test_size: float = 0.0
        self._random_state: int | None = None

        # either x_test or x_train will be set depending on tts
        self._predictions: np.ndarray | None = None
        # clamped predictions
        self._c_predictions: np.ndarray | None = None

        # only exist for testing models
        self._train_predictions: np.ndarray | None = None  # x_train predictions

    @abstractmethod
    def train_and_fit(self, tts:float = 0.0, random_state=42, **kwargs):
        """
        IMPLEMENTATION:
            - Call super() on this method.
            - Implement this method to: use either train_test_split() or train final model depending on if tts > 0.
            - Call self._fit() after training model.

        :param random_state:
        :param tts:     Train test split ratio
        :param kwargs:  Parameters to train the model  (e.g.  {random_state=42, hidden_layer_sizes=(100, 100), ...} )
        """
        if tts > 0:
            self._random_state = random_state
            self._test_size = tts
            self._x["train"], self._x["test"], self._y["train"], self._y["test"] = (
                train_test_split(self._x["full"], self._y["full"], test_size=tts, random_state=random_state))

    def _fit(self):
        """
        IMPLEMENTATION: call this method after training model in train_and_fit()
        """
        if isinstance(self._x["train"], pd.DataFrame):
            self.model.fit(self._x["train"], self._y["train"])
        else:
            self.model.fit(self._x["full"], self._y["full"])

    @abstractmethod
    def _score(self, folds: int=5):
        """
        After training, this method will be called calculate scores to add to self._scores
        """
        ts_cv = TimeSeriesSplit(n_splits=folds)

        cv_scores = cross_val_score(
            self.model,
            self._x["train"],
            self._y["train"].values.ravel(),
            scoring="r2",
            cv=ts_cv,
        )

        self._scores = {
            "R2": self.model.score(self._x["test"], self._y["test"]),
            "CV R2": np.mean(cv_scores),
            "RMSE": mean_squared_error(y_true=self._y["test"], y_pred=self._predictions) ** 0.5,
            "RMSE Clamped": mean_squared_error(y_true=self._y["test"], y_pred=self._c_predictions) ** 0.5,
            "MAE": mean_absolute_error(y_true=self._y["test"], y_pred=self._predictions),
            "CI": self._get_bootstrap(r2_score)
        }

    def _get_bootstrap(self, score_func: Callable, n_resamples: int = 1000) -> tuple[float, float]:
        res =  bootstrap(
            (np.ravel(self._y["test"].values), self._predictions),
            score_func,
            vectorized=False,
            paired=True,
            n_resamples=n_resamples,
            method='percentile',
            random_state=self._random_state)
        return res.confidence_interval.low, res.confidence_interval.high

    def predict(self, x: pd.DataFrame=None) -> np.ndarray:
        """
        Predict using parameter X or full dataset (if final model), or X_test (if testing model).

        :param x:  (optional) X to predict with
        :return:    predictions
        """
        if self.model is None:
            raise ValueError("Model has not been trained!")

        # testing models
        if self._test_size:
            self._predictions = self.model.predict(self._x["test"])
            self._clamp_predictions()
            self._score()

        # final models
        else:
            x = self._x["full"] if x is None else x
            self._predictions = self.model.predict(x)
            self._clamp_predictions()

        return self._predictions

    @abstractmethod
    def plot(self):
        pass

    def _clamp_predictions(self):
        """
        Optionally, call this method after setting self.predictions.

        Sets self._c_predictions to predictions clamped to 0 if elevation is 0.
        """
        index = self._y["test"].index if self._test_size else self._x["full"].index
        y_pred = pd.Series(self._predictions, index=index)

        # reindex elevation data to match predictions  (sometimes getting multiple timestamps?? not sure why)
        elev = self._elevation_df
        if elev.index.has_duplicates:
            # take the last value for each duplicated timestamp  (seems to only affect 1 or 2)
            elev = elev[~elev.index.duplicated(keep="last")]
        elev = elev.reindex(y_pred.index)

        # create and apply mask
        mask = (elev <= 0).fillna(False).to_numpy()
        y_pred.iloc[mask] = 0

        # modify predictions
        self._c_predictions = y_pred.to_numpy()

    def print_scores(self) -> None:
        # get the longest string for nice formatting
        longest_name = 0
        longest_score = 0
        for sn, sv in self._scores.items():
            if len(sn) > longest_name:
                longest_name = len(sn)

            # Unpack and format if it's a list or array/tuple
            if isinstance(sv, (list, np.ndarray, tuple)):
                val_str = ", ".join(str(round(x, 3)) for x in sv)
            else:
                val_str = str(round(sv, 3))

            if (longer_score := len(val_str)) > longest_score:
                longest_score = longer_score

        # print scores
        for sn, sv in self._scores.items():
            if isinstance(sv, (list, np.ndarray, tuple)):
                val_str = ", ".join(str(round(x, 3)) for x in sv)
            else:
                val_str = str(round(sv, 3))

            print(f"{sn:<{longest_name + 1}} | {val_str:>{longest_score + 1}}")

    def print_parameters(self) -> None:
        # get the longest string for nice formatting
        longest_name = 0
        longest_val = 0
        for pn, pv in self._parameters.items():
            if len(pn) > longest_name:
                longest_name = len(pn)

            # Unpack and format if it's a list or array
            if isinstance(pv, (list, np.ndarray, tuple)):
                val_str = ", ".join(str(round(x, 3)) for x in pv)
            else:
                val_str = str(round(pv, 3)) if isinstance(pv, (int, float)) else str(pv)

            if (longer_val := len(val_str)) > longest_val:
                longest_val = longer_val

        # print parameters
        for pn, pv in self._parameters.items():
            if isinstance(pv, (list, np.ndarray, tuple)):
                val_str = ", ".join(str(round(x, 3)) for x in pv)
            else:
                val_str = str(round(pv, 3)) if isinstance(pv, (int, float)) else str(pv)

            print(f"{pn:<{longest_name + 1}} | {val_str}")

    def get_scores(self) -> dict[str, float]:
        return self._scores

    def get_parameters(self) -> dict[str, any]:
        return self._parameters
