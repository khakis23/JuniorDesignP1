import numpy as np
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from Data.ModelData.ModelData import ModelData
from Models.src.IModel import IModel
from util.util import plot_daily_solar_output


class MLPRegression(IModel):

    def __init__(self, features: list[str], data: ModelData):
        super().__init__(features, data)

    def train_and_fit(self, random_state=42, **kwargs):
        super().train_and_fit(kwargs.get("tts", 0.0), random_state)
        kwargs.pop("tts", None)

        # train model
        self.model = make_pipeline(
            StandardScaler(),
            MLPRegressor(random_state=random_state, **kwargs)
        )
        # fit train or full data depending on tts
        self._fit()

    def _score(self, folds: int = 5):
        super()._score()

        mlp = self.model.named_steps["mlpregressor"]
        self._parameters = {
            "Test Size": self._test_size,
            "Iterations": mlp.n_iter_,
            "Loss": mlp.loss_,
            "Hidden Layers": mlp.hidden_layer_sizes,
            "Hidden Activation": mlp.activation,
            "Output Activation": mlp.out_activation_,
            "Alpha": mlp.alpha,
        }

    def plot(self):
        # testing model
        if self._test_size:
            display_features = {
                "R2": round(self._scores["R2"], 3),
                "CI": f"[{round(self._scores['CI'][0], 2)} : {round(self._scores['CI'][1], 2)}]",
                "RMSE": round(self._scores["RMSE Clamped"], 2),
                "Iters": self._parameters["Iterations"],
                "Layers": self._parameters["Hidden Layers"],
            }

            plot_daily_solar_output(
                self._c_predictions,
                self._y["test"],
                "MLP Regression",
                display_features,
            )

        # final model
        else:
            plot_daily_solar_output(
                self._c_predictions,
                self._y["full"],
                "MLP Regression",
            )