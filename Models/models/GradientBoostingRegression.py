import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from Data.ModelData.ModelData import ModelData
from Models.src.IModel import IModel
from util.plots import plot_daily_solar_output


class GradientBoostingRegression(IModel):

    def __init__(self, features: list[str], data: ModelData):
        super().__init__(features, data)

    def train_and_fit(self, random_state=42, **kwargs):
        super().train_and_fit(kwargs.get("tts", 0.0), random_state)
        kwargs.pop("tts", None)

        # train model
        self.model = make_pipeline(
            StandardScaler(),
            GradientBoostingRegressor(random_state=random_state, **kwargs)
        )
        # fit train or full data depending on tts
        self._fit()

    def _score(self, folds: int = 5):
        super()._score()

        gb = self.model.named_steps["gradientboostingregressor"]
        self._parameters = {
            "Test Size": self._test_size,
            "Estimators": gb.n_estimators,
            "Learning Rate": gb.learning_rate,
            "Max Depth": gb.max_depth,
            "Loss": gb.loss,
            "Subsample": gb.subsample,
            "Min Samples Split": gb.min_samples_split,
            "Min Samples Leaf": gb.min_samples_leaf,
            "Validation Fraction": gb.validation_fraction,
            "Early Stopping": gb.n_iter_no_change,
            "Tolerance": gb.tol,
        }

    def plot(self):
        # testing model
        if self._test_size:

            ci = self._scores.get("CI", (0.0, 0.0))
            ci_display = f"[{round(ci[0], 2)} : {round(ci[1], 2)}]" if isinstance(ci, (list, tuple)) else str(round(ci, 2))

            display_features = {
                "R2": round(self._scores["R2"], 3),
                "CI": ci_display,
                "RMSE": round(self._scores["RMSE Clamped"], 2),
                "Estimators": self._parameters["Estimators"],
                "LR": self._parameters["Learning Rate"]
            }

            plot_daily_solar_output(
                self._c_predictions,
                self._y["test"],
                "Gradient Boosting Regression",
                display_features,
            )

        # final model
        else:
            plot_daily_solar_output(
                self._c_predictions,
                self._y["full"],
                "Gradient Boosting Regression",
            )