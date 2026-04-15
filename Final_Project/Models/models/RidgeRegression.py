import numpy as np
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from Data.ModelData.ModelData import ModelData
from Models.src.ModelEval import ModelEval
from Models.src.IModel import IModel
from util.util import plot_daily_solar_output


class RidgeRegression(IModel):

    def __init__(self, features: list[str], data: ModelData):
        super().__init__(features, data)

    def train_and_fit(self, random_state=42, **kwargs):
        super().train_and_fit(kwargs.get("tts", 0.0), random_state)
        kwargs.pop("tts", None)

        # train model
        self.model = make_pipeline(
            StandardScaler(),
            RidgeCV(**kwargs)
        )
        # fit train or full data depending on tts
        self._fit()

    def _score(self, folds: int = 5):
        self._scores = {
            "R2": self.model.score(self._x["test"], self._y["test"]),
            "RMSE": mean_squared_error(y_true=self._y["test"], y_pred=self._predictions) ** 0.5,
            "RMSE Clamped": mean_squared_error(y_true=self._y["test"], y_pred=self._c_predictions) ** 0.5,
            "MAE": mean_absolute_error(y_true=self._y["test"], y_pred=self._predictions),
            "CI": self._get_bootstrap(r2_score)
        }
        self._parameters = {
            "Alpha": self.model.named_steps["ridgecv"].alpha_,
            "Test Size": self._test_size,
            "Coefs": self.model.named_steps["ridgecv"].coef_,
            "Intercept": self.model.named_steps["ridgecv"].intercept_,
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
                "Alpha": round(self._parameters["Alpha"], 3),
            }

            plot_daily_solar_output(
                self._c_predictions,
                self._y["test"],
                "Ridge Regression",
                display_features,
            )

        # final model
        else:
            plot_daily_solar_output(
                self._c_predictions,
                self._y["full"],
                "Ridge Regression",
            )