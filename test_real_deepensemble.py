import numpy as np
import pandas as pd

from Models.models.DeepEnsemble import DeepEnsemble


class FakeData:
    def __init__(self):
        self.weather = pd.DataFrame({
            "sunelevation": [1, 2, 3]
        })
        self.energy = np.array([10.0, 20.0, 30.0])
        self.features = pd.DataFrame({
            "a": [1.0, 2.0, 3.0],
            "b": [4.0, 5.0, 6.0],
        })


class FakeModel:
    def __init__(self, features, data):
        self.features = features
        self.data = data
        self.seed = 0

    def train_and_fit(self, random_state=42, tts=0.0, **kwargs):
        self.seed = random_state

    def predict(self, X):
        X = np.asarray(X)
        return np.sum(X, axis=1) + self.seed * 0.01

    def predict_mean_variance(self, X):
        X = np.asarray(X)
        mu = np.sum(X, axis=1) + self.seed * 0.01
        var = np.ones(len(X)) * (0.1 + self.seed * 0.001)
        return mu, var

    def get_scores(self):
        return {"R2": 0.9, "RMSE": 1.0}


data = FakeData()
features = ["a", "b"]

ensemble = DeepEnsemble(
    features=features,
    data=data,
    base_model_class=FakeModel,
    n_models=3
)

ensemble.train_and_fit(random_state=42)

X_test = np.array([
    [1.0, 2.0],
    [3.0, 4.0],
    [5.0, 6.0]
])

results = ensemble.predict_with_uncertainty(X_test)

print("mean:", results["mean"])
print("std:", results["std"])
print("total_var:", results["total_var"])
print("epistemic_var:", results["epistemic_var"])
print("aleatoric_var:", results["aleatoric_var"])
print("member_means shape:", results["member_means"].shape)
print("member_vars shape:", results["member_vars"].shape)