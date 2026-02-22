from abc import ABC

from models.IModel import *


class NeuralNetRegression(IModel):

    def __init__(self, features: list[str], data: ModelData=ModelData(), final=False, **kwargs):
        super().__init__(features, data, final, **kwargs)

    def _train_and_fit(self, **kwargs):
        # handle optional arguments
        tts = {}
        if "random_state" in kwargs:
            tts["random_state"] = kwargs["random_state"]
        if "test_size" in kwargs:
            tts["test_size"] = kwargs["test_size"]
        else:
            tts["test_size"] = 0.2
        self.test_size = tts["test_size"]

        self.alpha = kwargs["alpha"] if "alpha" in kwargs else .0001
        self.hidden_layers = kwargs["hidden_layers"] if "hidden_layers" in kwargs else (64, 32)
        self.max_iter = kwargs["max_iter"] if "max_iter" in kwargs else 1000

        # train test split
        self.x_train, self.x_test, self.y_train, self.y_test \
            = train_test_split(self._x, self._y, **tts)

        # Train Neural Network
        mlp = MLPRegressor(
            hidden_layer_sizes=self.hidden_layers,
            solver='adam',
            activation='relu',
            max_iter=self.max_iter,
            random_state=tts.get("random_state"),
            alpha=self.alpha
        )
        self.model = make_pipeline(StandardScaler(), mlp)
        self.model.fit(self.x_train, np.ravel(self.y_train))

        # Save reference to the raw MLP object for printing stats
        self.mlp_model = self.model.named_steps["mlpregressor"]

        # predict
        self.predictions = self.model.predict(self.x_test)

    def print_results(self):
        print(f"MSE: {self.mse:.3f}")
        print(f"RMSE: {self.rmse:.3f}")
        print(f"RMSE Clamped: {self.rmse_clamped:.3f}")
        print(f"R2: {self.r2:.3f}")
        print(f"Test Size: {self.test_size:.1%}")
        print(f"Alpha: {self.alpha:.3f}")
        print(f"Hidden Layers: {self.hidden_layers}")
        print(f"Max Iterations: {self.max_iter}")
        print(f"Epochs to converge: {self.mlp_model.n_iter_}\n")

    def _evaluate(self):
        # normally scoring
        self.mse = mean_squared_error(self.y_test, self.predictions)
        self.rmse = self.mse ** 0.5
        self.r2 = self.model.score(self.x_test, self.y_test)

        self._clamp_predictions()

        # scoring after clamping
        self.rmse_clamped = mean_squared_error(self.y_test, self.predictions) ** 0.5

    def _final_train_and_fit(self, **kwargs):
        pass


class NeuralNetEval(IModelEval):

    def __init__(self, models: list[NeuralNetRegression]):
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
    model = NeuralNetRegression(["temp",  "cloudcover",  "solarradiation",  "hcos",  "dtemp",  "dsolarradiation"])
    model.print_results()
    model.plot()