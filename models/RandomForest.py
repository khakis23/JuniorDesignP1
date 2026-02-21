from models.IModel import *
from sklearn.ensemble import RandomForestRegressor


class RandomForestModel(IModel):

    def __init__(self, features: list[str], **kwargs):
        super().__init__(features)

    def _train_and_fit(self, **kwargs):
        # handle optional arguments (match RidgeRegression style)
        tts = {}
        if "random_state" in kwargs:
            tts["random_state"] = kwargs["random_state"]
        if "test_size" in kwargs:
            tts["test_size"] = kwargs["test_size"]
        else:
            tts["test_size"] = 0.2  # default

        # train-test split
        self.x_train, self.x_test, self.y_train, self.y_test = \
            train_test_split(self._x, self._y, **tts)

        # --- Random Forest hyperparams (overridable like kwargs) ---
        n_estimators = kwargs.get("n_estimators", 200)
        max_depth = kwargs.get("max_depth", 15)
        min_samples_leaf = kwargs.get("min_samples_leaf", 3)
        min_samples_split = kwargs.get("min_samples_split", 2)
        max_features = kwargs.get("max_features", "sqrt")  # matches RF notes (random subset of features)
        bootstrap = kwargs.get("bootstrap", True)

        self.model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            min_samples_split=min_samples_split,
            max_features=max_features,
            bootstrap=bootstrap,
            random_state=kwargs.get("random_state", 42),
            n_jobs=-1
        )

        # --- FIX: sklearn wants y as (n,), not (n,1) ---
        y_train_1d = self.y_train
        if isinstance(y_train_1d, pd.DataFrame):
            y_train_1d = y_train_1d.values.ravel()
        elif isinstance(y_train_1d, pd.Series):
            y_train_1d = y_train_1d.values

        # train
        self.model.fit(self.x_train, y_train_1d)

        # predict
        self.predictions = self.model.predict(self.x_test)
        # NOTE: predictions will be clamped by _evaluate()

    def _evaluate(self):
        # raw scoring
        self.mse = mean_squared_error(self.y_test, self.predictions)
        self.rmse = self.mse ** 0.5
        self.r2 = r2_score(self.y_test, self.predictions)

        # clamp nighttime predictions (copied from RidgeRegression)
        self._clamp_predictions()

        # scoring after clamping
        self.rmse_clamped = mean_squared_error(self.y_test, self.predictions) ** 0.5

    def _clamp_predictions(self):
        y_pred = pd.Series(self.predictions, index=self.y_test.index)

        # reindex elevation data to match predictions (handles duplicated timestamps)
        elev = self.elevation_df
        if elev.index.has_duplicates:
            elev = elev[~elev.index.duplicated(keep="last")]
        elev = elev.reindex(y_pred.index)

        # create and apply mask
        mask = (elev <= 0).fillna(False).to_numpy()
        y_pred.iloc[mask] = 0

        # modify predictions
        self.predictions = y_pred.to_numpy()

    def print_results(self):
        print(f"MSE: {self.mse}")
        print(f"RMSE: {self.rmse}")
        print(f"RMSE Clamped: {self.rmse_clamped}")
        print(f"R2: {self.r2}")


class RandomForestEval(IModelEval):

    def __init__(self, models: list[RandomForestModel]):
        super().__init__(models)
        self.r2_idx: int
        self.rmse_raw_idx: int
        self.rmse_clamped_idx: int

    def evaluate(self):
        best_r2_raw = -np.inf
        best_rmse_raw = np.inf
        best_rmse_clamped = np.inf

        for i, model in enumerate(self.models):
            if model.r2 > best_r2_raw:
                best_r2_raw = model.r2
                self.r2_idx = i

            if model.rmse < best_rmse_raw:
                best_rmse_raw = model.rmse
                self.rmse_raw_idx = i

            if model.rmse_clamped < best_rmse_clamped:
                best_rmse_clamped = model.rmse_clamped
                self.rmse_clamped_idx = i

        self.best_models.append(self.models[self.r2_idx])
        if self.r2_idx != self.rmse_raw_idx:
            self.best_models.append(self.models[self.rmse_raw_idx])
        if self.r2_idx != self.rmse_clamped_idx and self.rmse_raw_idx != self.rmse_clamped_idx:
            self.best_models.append(self.models[self.rmse_clamped_idx])


if __name__ == "__main__":
    tests = [
        RandomForestModel(
            ["temp", "solarradiation", "sunelevation", "cloudcover", "sunazimuth", "solarenergy"],
            random_state=42,
            n_estimators=300,
            max_depth=15,
            min_samples_leaf=3
        ),
        RandomForestModel(
            ["temp", "sunelevation", "cloudcover", "sunazimuth", "solarenergy", "hsin", "hcos", "dsin", "dcos"],
            random_state=42,
            n_estimators=300,
            max_depth=15,
            min_samples_leaf=3
        ),
    ]

    ev = RandomForestEval(tests)
    ev.evaluate()
    ev.display_best()