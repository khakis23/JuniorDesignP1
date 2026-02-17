from sklearn.metrics import mean_squared_error

from models.Data import *


class RidgeRegression:

    def __init__(self, features: list[str], test_size=0.2, **kwargs):
        self.features = features

        ### setup data ###
        md = ModelData()
        self._y = md.energy
        x = md.weather

        # add cyclic time features
        hour = x.index.hour
        x["hsin"] = np.sin(2 * np.pi * hour / 24)
        x["hcos"] = np.cos(2 * np.pi * hour / 24)

        day = x.index.dayofyear
        x["dsin"] = np.sin(2 * np.pi * day / 365.25)   # cant forget leap year
        x["dcos"] = np.cos(2 * np.pi * day / 365.25)

        # data including only selcted features
        self._x = md.weather[features]

        # train test split
        tts = {}
        if "random_state" in kwargs:
            tts = {"random_state": kwargs["random_state"]}

        self.x_train, self.x_test, self.y_train, self.y_test \
            = train_test_split(self._x, self._y, test_size=test_size, **tts)

        # train model
        self.model = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-3, 3, 50)))   # cross validation
        self.model.fit(self.x_train, self.y_train)
        self.ridge = self.model.named_steps["ridgecv"]

        # predict
        self.predictions = self.model.predict(self.x_test)
        # clamp predictions since cannot have negative energy
        self.predictions = self.predictions.clip(min=0)

        # evalute
        self.mse = mean_squared_error(self.y_test, self.predictions)
        self.rmse = self.mse ** 0.5

    def print_results(self):
        print(f"MSE: {self.mse}")
        print(f"RMSE: {self.rmse}")
        print(f"R2: {self.model.score(self.x_test, self.y_test)}")
        print(f"Alpha: {self.ridge.alpha_}")
        print(f"coef: ", *self.ridge.coef_, sep=", ")
        print("Intercept: ", self.ridge.intercept_[0], "\n")

    def plot_predictions(self):
        # organize data into an averaged day
        yt_plot = self.y_test.groupby(self.y_test.index.hour).mean()
        yp_plot = pd.Series(self.predictions, index=self.y_test.index)
        yp_plot = yp_plot.groupby(yp_plot.index.hour).mean()

        # plot
        fig, ax = plt.subplots()
        ax.plot(yt_plot.index, yt_plot["power"], label="Actual")
        ax.plot(yp_plot.index, yp_plot, label="Predicted")
        ax.legend()
        plt.show()



if __name__ == "__main__":
    tests = [
        RidgeRegression(["temp", "solarradiation", "sunelevation", "cloudcover", "sunazimuth", "solarenergy"], random_state=42),
        RidgeRegression(["temp", "sunelevation", "cloudcover", "sunazimuth", "solarenergy"], random_state=42),
        RidgeRegression(["temp", "sunelevation", "cloudcover", "sunazimuth", "solarenergy", "hsin", "hcos", "dsin", "dcos"], random_state=42),
    ]

    for test in tests:
        test.print_results()
        test.plot_predictions()





