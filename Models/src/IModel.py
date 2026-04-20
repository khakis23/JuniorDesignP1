import random
from abc import ABC, abstractmethod
from pathlib import Path
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


def _format_param_value(value: any) -> str:
    """
    Helper function to format hyperparameter values for printing.
    Rounds numbers to 4 decimal places and cleanly joins iterables.
    """
    if isinstance(value, (list, np.ndarray, tuple)):
        # only attempt to round if the element is actually a number
        return ", ".join(
            str(round(x, 4)) if isinstance(x, (int, float, np.number)) else str(x)
            for x in value
        )

    # Handle single values
    if isinstance(value, (int, float, np.number)) and not isinstance(value, bool):
        return str(round(value, 4))

    return str(value)


class IModel(ABC):
    """
    Abstract Base Class for all machine learning models in the ModelMaker pipeline.

    This class provides a unified interface for data splitting, training orchestration,
    prediction routing, and metric scoring. By inheriting from `IModel`, any custom
    model (Sklearn, TensorFlow, PyTorch, or Custom Ensembles) can seamlessly plug into
    the `Trainer` and `ModelMaker` grid search infrastructure.

    IMPLEMENTATION GUIDE FOR SUBCLASSES:
    To create a new model, you must:
        1. Inherit from `IModel`.
        2. Define `self.model` (the actual underlying ML model) in your `train_and_fit` method.
        3. Implement the abstract `train_and_fit(...)` method.
           - Step A: MUST call `super().train_and_fit(tts, random_state)` to handle data splitting.
           - Step B: Extract your specific hyperparameters from `**kwargs` and log them to `self._parameters`.
           - Step C: Initialize your underlying model to `self.model`.
           - Step D: Call `self._fit()` (if your model has an sklearn-style `.fit()` method) OR manually
                     fit your model using `self._x["train"]` and `self._y["train"]`.
        4. (Optional) Override `predict()` if your model requires specialized tensor conversions
           or outputs multiple elements (like predicting mean AND variance).
    """

    def __init__(self, features: list[str], targets: list[str], data: pd.DataFrame, data_test: pd.DataFrame = None,
                 plot_func: Callable = None):
        """
        :param features:    List of string column names representing the independent variables (X).
        :param targets:     List of string column names representing the dependent variables (Y).
        :param data:        The primary Pandas DataFrame containing both features and targets.
        :param data_test:   (Optional) A secondary DataFrame used explicitly for testing. If provided,
                            the model bypasses `train_test_split` and uses this for evaluation.
        :param plot_func:   A callable function that takes the model and data dictionaries
                            to generate visual plots.
        """
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
        Abstract method where the underlying ML model is instantiated and trained.

        SUBCLASS IMPLEMENTATION REQUIREMENTS:
            1. Call `super().train_and_fit(tts, random_state)` first to execute the data splitting logic.
            2. Build your model using the hyperparameter instructions found in `kwargs`.
            3. Save those hyperparameters to `self._parameters` for logging.
            4. Call `self._fit()` to execute the training sequence.

        :param tts:             Train-test split ratio (e.g., 0.2 for 20% test data). If 0.0, trains on full dataset.
        :param random_state:    Integer for reproducible data splitting and model initialization.
        :param kwargs:          Dynamic dictionary of hyperparameters passed down from ModelMaker/Trainer
                                (e.g., {learning_rate=0.001, hidden_layer_sizes=(100, 100)}).
        """
        rand_or_none = {}
        if random_state is not None:
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
        Executes the `.fit()` command on the underlying `self.model`.

        IMPLEMENTATION: Call this method as the final step inside your overridden `train_and_fit()` method.
        Assumes `self.model` follows the scikit-learn API structure of `fit(X, Y)`. If your custom
        model (e.g., raw TensorFlow) requires a custom training loop, do not call this; implement
        the fitting logic directly in your `train_and_fit()`.
        """
        # Fit on train if TTS occurred OR if data was pre-split
        if self._test_size > 0 or self._pre_split:
            self.model.fit(self._x["train"], self._y["train"])
        else:
            self.model.fit(self._x["full"], self._y["full"])

    def _score(self, folds: int = 5):
        """
        Evaluates the model's predictions against the test set and populates `self._scores`.

        Calculates standard regression metrics (R2, RMSE, MAE) and a 95% Bootstrap Confidence Interval.
        If your custom model introduces new metrics (like Uncertainty Quantification), you should call
        `super()._score()` first, and then append your custom metrics to the `self._scores` dictionary.

        :param folds: Integer defining the number of splits for TimeSeries Cross-Validation.
        """
        cv_r2 = np.nan

        try:
            ts_cv = TimeSeriesSplit(n_splits=folds)
            cv_scores = cross_val_score(   # TODO broken with non-sklearn models!
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
        """
        Internal helper method to calculate a confidence interval for the primary scoring metric.

        :param score_func:  Callable scoring function (e.g., `sklearn.metrics.r2_score`).
        :param n_resamples: Integer dictating how many bootstrap iterations to perform.
        :return:            Tuple containing the (lower_bound, upper_bound) of the confidence interval.
        """
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
        Generates predictions and handles internal routing for evaluation.

        If the model was trained with a test split (`tts > 0` or `data_test` provided), this method
        automatically evaluates on the internal test set and triggers `self._score()`. If it was trained
        on the full dataset (final model), it predicts on the full dataset or the optionally provided `x`.

        :param x:  (Optional) External Pandas DataFrame to generate predictions for.
        :return:   Numpy array containing the predictions.
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
        """
        Prints the model's evaluation metrics (R2, RMSE, etc.) in a formatted, aligned table.
        """
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
        """
        Prints the model's hyperparameter configuration in a formatted, aligned table.
        Subclasses must populate `self._parameters` during `train_and_fit` for this to work.
        """
        if not self._parameters:
            return

        print("\n————— Model Parameters —————")

        longest_name = max((len(pn) for pn in self._parameters.keys()), default=0)

        for pn, pv in self._parameters.items():
            val_str = _format_param_value(pv)
            print(f"{pn:<{longest_name + 1}} | {val_str}")

    def write_scores_to_file(self, path: Path | str, append: bool = True) -> None:
        """
        Writes the model's evaluation metrics to a file in an aligned table format.
        """
        # Convert string path to Path object if necessary
        path = Path(path)
        mode = "a" if append else "w"

        longest_name = max((len(sn) for sn in self._scores.keys()), default=0)
        longest_score = 0

        # Calculate max score string length for alignment
        formatted_scores = {}
        for sn, sv in self._scores.items():
            if isinstance(sv, (list, np.ndarray, tuple)):
                val_str = ", ".join(str(round(x, 3)) for x in sv)
            else:
                val_str = str(round(sv, 3))
            formatted_scores[sn] = val_str
            longest_score = max(longest_score, len(val_str))

        with open(path, mode) as f:
            f.write("\n\n————— Model Scores —————\n")
            for sn, val_str in formatted_scores.items():
                f.write(f"{sn:<{longest_name + 1}} | {val_str:>{longest_score + 1}}\n")

    def write_parameters_to_file(self, path: Path | str, append: bool = True) -> None:
        """
        Writes model features and hyperparameter configuration to a file.
        """
        # Safety check: if there's nothing to write, just exit
        if not self._parameters and not (hasattr(self, 'features') and self.features):
            return

        path = Path(path)
        mode = "a" if append else "w"

        with open(path, mode) as f:
            # Write Features Section
            if hasattr(self, 'features') and self.features:
                f.write("\n————— Model Features —————\n")
                f.write("  ".join(str(feat) for feat in self.features) + "\n")

            # Write Parameters Section
            if self._parameters:
                f.write("\n————— Model Parameters —————\n")

                longest_name = max((len(pn) for pn in self._parameters.keys()), default=0)

                for pn, pv in self._parameters.items():
                    val_str = _format_param_value(pv)
                    f.write(f"{pn:<{longest_name + 1}} | {val_str}\n")

    def get_scores(self) -> dict[str, float]:
        return self._scores

    def get_parameters(self) -> dict[str, any]:
        """hyperparameter configuration."""
        return self._parameters

    def get_features(self) -> list[str]:
        return self.features

    def get_targets(self) -> list[str]:
        """target column names the model is predicting."""
        return self.targets

    def set_plot(self, func: Callable):
        """
        Binds an external plotting function to this specific model instance.

        :param func: A callable function matching the signature `func(model, x_dict, y_dict, **kwargs)`.
        """
        self._plot_func = func

    def plot(self, **kwargs):
        """
        Executes the bound plotting function.
        Will print an error message if `set_plot` has not been called prior to execution.
        """
        if not self._plot_func:
            print("No plot function set. Call set_plot() first.")
            return

        self._plot_func(self, self._x, self._y, **kwargs)