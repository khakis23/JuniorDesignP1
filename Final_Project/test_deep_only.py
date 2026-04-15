import numpy as np


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


class DeepEnsemble:
    def __init__(self, features, data, base_model_class=None, n_models=5):
        self.features = features
        self.data = data
        self.base_model_class = base_model_class
        self.n_models = n_models

        self.models = []
        self.member_scores = []

        self.mean_prediction = None
        self.prediction_std = None
        self.epistemic_var = None
        self.aleatoric_var = None
        self.total_var = None

        self.member_means = None
        self.member_vars = None

    def train_and_fit(self, random_state=42, **kwargs):
        if self.base_model_class is None:
            raise ValueError("DeepEnsemble requires base_model_class")

        self.models = []
        self.member_scores = []

        for i in range(self.n_models):
            member_seed = random_state + i
            model = self.base_model_class(self.features, self.data)
            model.train_and_fit(random_state=member_seed, **kwargs)
            self.models.append(model)
            self.member_scores.append(model.get_scores())

    def predict(self, X):
        mus = []
        vars_ = []

        for model in self.models:
            mu_m, var_m = model.predict_mean_variance(X)
            mus.append(np.asarray(mu_m).reshape(-1))
            vars_.append(np.asarray(var_m).reshape(-1))

        mus = np.vstack(mus)
        vars_ = np.vstack(vars_)

        self.member_means = mus
        self.member_vars = vars_

        self.mean_prediction = np.mean(mus, axis=0)
        self.epistemic_var = np.var(mus, axis=0)
        self.aleatoric_var = np.mean(vars_, axis=0)
        self.total_var = self.epistemic_var + self.aleatoric_var
        self.prediction_std = np.sqrt(self.total_var)

        return self.mean_prediction

    def predict_with_uncertainty(self, X):
        mean_pred = self.predict(X)
        return {
            "mean": mean_pred,
            "std": self.prediction_std,
            "total_var": self.total_var,
            "epistemic_var": self.epistemic_var,
            "aleatoric_var": self.aleatoric_var,
            "member_means": self.member_means,
            "member_vars": self.member_vars,
        }


features = ["a", "b"]
data = None

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