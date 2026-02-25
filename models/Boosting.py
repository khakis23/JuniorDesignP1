from numpy.ma.core import ravel
from models.IModel import *


class GradientBoostRegress(IModel):
    def __init__(self, features: list[str], data: ModelData=ModelData(), **kwargs):
        super().__init__(features, data, **kwargs)

    def _train_and_fit(self, **kwargs):
        # handle optional arguments
        random_state = {}
        if "random_state" in kwargs:
            random_state["random_state"] = kwargs["random_state"]
        else:
            random_state["random_state"] = 0

        if "n_estimators" in kwargs:
            self.n_estimators = kwargs["n_estimators"]
        else:
            self.n_estimators = 100
        if "max_depth" in kwargs:
            self.depths = kwargs["max_depth"]
        else:
            self.depths = 3  # default

        # train test split
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self._x, self._y, **random_state)

        # train model
        self.model = GradientBoostingRegressor(learning_rate=0.1, n_estimators=self.n_estimators, max_depth=self.depths,
                                               random_state=random_state["random_state"])
        self.model.fit(self.x_train, self.y_train.values.ravel())
        self.predictions = self.model.predict(self.x_test)
    
    def _evaluate(self):
        # normally scoring
        self.mse = mean_squared_error(self.y_test, self.predictions)
        self.rmse = self.mse ** 0.5
        self.r2 = self.model.score(self.x_test, self.y_test)

        self._clamp_predictions()

        # scoring after clamping
        self.rmse_clamped = mean_squared_error(self.y_test, self.predictions) ** 0.5

    def print_results(self):
        print(f"MSE: {self.mse}")
        print(f"RMSE: {self.rmse}")
        print(f"RMSE Clamped: {self.rmse_clamped}")
        print(f"R2: {self.r2}")
        print(f"Num Estimators: {self.n_estimators}")
        print(f"Depth: {self.depths}")

    # optionally override
    def _set_features(self):
        self._x = self._x[self.features]

    # optionally override
    def _final_train_and_fit(self, **kwargs):
        pass


class GradientBoostEval(IModelEval):
    def __init__(self, models: list[GradientBoostRegress]):
        super().__init__(models)
        # best model indices
        self.r2_idx: int
        self.rmse_raw_idx: int
        self.rmse_clamped_idx: int

    def evaluate(self):
        """
        This method implements parent method, finding model with best R2, RMSE, and RMSE Clamped.
        """
        best_r2_raw = 0
        best_rmse_raw = np.inf
        best_rmse_clamped = np.inf

        # find best models
        for i, model in enumerate(self.models):
            # R2 Raw
            if model.r2 > best_r2_raw:
                best_r2_raw = model.r2
                self.r2_idx = i

            # RMSE Raw
            if model.rmse < best_rmse_raw:
                best_rmse_raw = model.rmse
                self.rmse_raw_idx = i

            # RMSE Clamped
            if model.rmse_clamped < best_rmse_clamped:
                best_rmse_clamped = model.rmse_clamped
                self.rmse_clamped_idx = i

        # add best model(s) to list
        self.best_models.append(self.models[self.r2_idx])
        if self.r2_idx != self.rmse_raw_idx:
            self.best_models.append(self.models[self.rmse_raw_idx])
        if self.r2_idx != self.rmse_clamped_idx and self.rmse_raw_idx != self.rmse_clamped_idx:
            self.best_models.append(self.models[self.rmse_clamped_idx])


    def _clamp_predictions(self):
        y_pred = pd.Series(self.predictions, index=self.y_test.index)

        # reindex elevation data to match predictions  (sometimes getting multiple timestamps?? not sure why)
        elev = self.elevation_df
        if elev.index.has_duplicates:
            # take the last value for each duplicated timestamp  (seems to only affect 1 or 2)
            elev = elev[~elev.index.duplicated(keep="last")]
        elev = elev.reindex(y_pred.index)

        # create and apply mask
        mask = (elev <= 0).fillna(False).to_numpy()
        y_pred.iloc[mask] = 0

        # modify predictions
        self.predictions = y_pred.to_numpy()
    

y = ["hcos", "cloudcover", "temp"]
if __name__ == "__main__":
    gb = GradientBoostRegress(y, max_depth=3, n_estimators=100, random_state=0)
    gb.print_results()
