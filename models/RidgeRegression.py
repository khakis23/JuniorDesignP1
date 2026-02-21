from models.IModel import *


class RidgeRegression(IModel):

    def __init__(self, features: list[str], data: ModelData=ModelData(), final=False, **kwargs):
        super().__init__(features, data, final, **kwargs)

    def _train_and_fit(self, **kwargs):
        # handle optional arguemnts
        tts = {}
        if "random_state" in kwargs:
            tts["random_state"] = kwargs["random_state"]
        if "test_size" in kwargs:
            self.test_size = kwargs["test_size"]
        else:
            self.test_size = 0.2  ### default test size
        tts["test_size"] = self.test_size

        # train test split
        self.x_train, self.x_test, self.y_train, self.y_test \
            = train_test_split(self._x, self._y, **tts)

        # train model
        self.model = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-3, 3, 50)))  # cross validation
        self.model.fit(self.x_train, self.y_train)
        self.ridge = self.model.named_steps["ridgecv"]

        # predict
        self.predictions = self.model.predict(self.x_test)
        # NOTE self.preditions is altered by _evalute to clamp to 0

    def _final_train_and_fit(self, **kwargs):
        try:
            alpha = kwargs.get("alpha")
        except KeyError:
            print("No alpha value provided! Using default alpha=1.")
            alpha = 1
        self.model = make_pipeline(StandardScaler(), Ridge(alpha=alpha))
        self.model.fit(self._x, self._y)

    def _evaluate(self):
        # normally scoring
        self.mse = mean_squared_error(self.y_test, self.predictions)
        self.rmse = self.mse ** 0.5
        self.r2 = self.model.score(self.x_test, self.y_test)

        self._clamp_predictions()

        # scoring after clamping
        self.rmse_clamped = mean_squared_error(self.y_test, self.predictions) ** 0.5

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

    def print_results(self):
        print(f"MSE: {self.mse:.3f}")
        print(f"RMSE: {self.rmse:.4f}")
        print(f"RMSE Clamped: {self.rmse_clamped:.4f}")
        print(f"R2: {self.r2:.4f}")
        print(f"Alpha: {self.ridge.alpha_:.4f}")
        print(f"Test Size: {self.test_size:.1%}")
        print("Coefs:", *np.round(self.ridge.coef_, 2), sep=", ")
        print(f"Intercept: {self.ridge.intercept_[0]:.2f}\n")


class RidgeRegEval(IModelEval):

    def __init__(self, models: list[RidgeRegression]):
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



if __name__ == "__main__":
    tests = [
        RidgeRegression(["temp", "solarradiation", "sunelevation", "cloudcover", "sunazimuth", "solarenergy"], random_state=42),
        RidgeRegression(["temp", "sunelevation", "cloudcover", "sunazimuth", "solarenergy"], random_state=42),
        RidgeRegression(["temp", "sunelevation", "cloudcover", "sunazimuth", "solarenergy", "hsin", "hcos", "dsin", "dcos"], random_state=42),
    ]

    for test in tests:
        test.print_results()
        test.plot()

