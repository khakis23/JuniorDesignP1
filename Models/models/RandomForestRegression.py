import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from Data.ModelData.ModelData import ModelData
from Models.src.IModel import IModel
from util.plots import plot_daily_solar_output


class RandomForestRegression(IModel):

    def __init__(self, features: list[str], data: ModelData):
        super().__init__(features, data)

    def train_and_fit(self, random_state=42, **kwargs):
        super().train_and_fit(kwargs.get("tts", 0.0), random_state)
        kwargs.pop("tts", None)

        # train model
        self.model = make_pipeline(
            StandardScaler(),
            RandomForestRegressor(random_state=random_state, **kwargs)
        )
        # fit train or full data depending on tts
        self._fit()

    def _score(self, folds: int = 5):
        super()._score()

        rf = self.model.named_steps["randomforestregressor"]
        self._parameters = {
            "Test Size": self._test_size,
            "Estimators": rf.n_estimators,
            "Max Depth": rf.max_depth,
            "Min Samples Split": rf.min_samples_split,
            "Min Samples Leaf": rf.min_samples_leaf,
            "Max Features": rf.max_features,
            "Bootstrap": rf.bootstrap,
            "Features Seen": rf.n_features_in_,
        }

    def plot(self):
        # testing model
        if self._test_size:
            display_features = {
                "R2": round(self._scores["R2"], 3),
                "CI": f"[{round(self._scores['CI'][0], 2)} : {round(self._scores['CI'][1], 2)}]",
                "RMSE": round(self._scores["RMSE Clamped"], 2),
                "Estimators": self._parameters["Estimators"],
                "Max Depth": self._parameters["Max Depth"] if self._parameters["Max Depth"] is not None else "None"
            }

            plot_daily_solar_output(
                self._c_predictions,
                self._y["test"],
                "Random Forest Regression",
                display_features,
            )

        # final model
        else:
            plot_daily_solar_output(
                self._c_predictions,
                self._y["full"],
                "Random Forest Regression",
            )