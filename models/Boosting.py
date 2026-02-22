from models.Data import ModelData
from models.IModel import *
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor

class GradientBoostRegress(IModel):
    def __init__(self, features: list[str], **kwargs):
        super().__init__(features)
    def _train_and_fit(self, **kwargs):
        # handle optional arguemnts
        tts = {}
        if "random_state" in kwargs:
            tts["random_state"] = kwargs["random_state"]
        if "test_size" in kwargs:
            tts["test_size"] = kwargs["test_size"]
        else:
            tts["test_size"] = 0.2   ### default test size

        # train test split
        self.x_train, self.x_test, self.y_train, self.y_test \
            = train_test_split(self._x, self._y, **tts)

        # train model
        self.model = make_pipeline(StandardScaler(), GradientBoostingRegressor(n_estimators=100, 
                                                                                max_depth=10,
                                                                                learning_rate=0.1,
                                                                                random_state=42))  # cross validation
        self.model.fit(self.x_train, self.y_train)
        self.gradBoost = self.model.named_steps["gradBoost"]
        self.predictions = self.model.predict(self.x_test)
    
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
        print(f"MSE: {self.mse}")
        print(f"RMSE: {self.rmse}")
        print(f"RMSE Clamped: {self.rmse_clamped}")
        print(f"R2: {self.r2}")
        # print(f"# of trees: {}")

# Note: Currently trying to match formatting and adding functionality to fit gradiant boosting
