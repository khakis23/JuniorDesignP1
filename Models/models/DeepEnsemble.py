from typing import Callable

import numpy as np
import pandas as pd
from Models.src.IModel import IModel
from Models.src.ProgressBar import ProgressBar, clear_last_line


class DeepEnsemble(IModel):
    """
    Deep Ensemble with full uncertainty decomposition.

    Each member model must implement:
        - train_and_fit(...)
        - predict(...)
        - predict_mean_variance(...)
        - get_scores()

    Full uncertainty used here:
        total_var = mean(member_var) + var(member_mean)
                  = aleatoric_var + epistemic_var
    """

    def __init__(self, features: list[str], targets: list[str], data: pd.DataFrame,
                 data_test: pd.DataFrame = None, plot_func: Callable=None):

        super().__init__(features, targets, data, data_test, plot_func)

        self.data = data
        self.data_test = data_test
        self.base_model_class: IModel | None = None
        self.n_models: int = 5

        self.models: list[IModel] = []
        self.member_scores = []

        self.mean_prediction = None
        self.prediction_std = None
        self.epistemic_var = None
        self.aleatoric_var = None
        self.total_var = None

        self.member_means = None
        self.member_vars = None

        self._ensemble_params = {}

        self.progress_bar = ProgressBar(warmup_gpu=False)  # gpu warmed in Trainer

    def train_and_fit(self, tts: float = 0.0, random_state=42, **kwargs):
        # Base class splitting
        super().train_and_fit(tts, random_state)

        # Allow passing class via kwargs or use the one from __init__
        self.base_model_class = kwargs.pop("base_model_class", self.base_model_class)
        self.n_models = kwargs.pop("n_models", self.n_models)

        self.base_model_class = self.base_model_class
        self.models = []
        self.member_scores = []

        # Store ensemble configuration for logging
        self._ensemble_params = {
            "base_model_class": getattr(self.base_model_class, "__name__", str(self.base_model_class)),
            "n_models": self.n_models,
            **kwargs,
        }

        # start progress bar
        print(f"\nTraining Deep Ensemble with {self.n_models} {self.base_model_class.__name__} models...", flush=True)
        self.progress_bar.set_max_steps(self.n_models)

        for i in range(self.n_models):
            # Advance seed to ensure diversity in initializations and training shuffles
            member_seed = None if random_state is None else random_state + i

            # Instantiate a fresh instance of the base model
            # Pass targets and data_test to support multi-output and pre-split
            model = self.base_model_class(self.features, self.targets, self.data, self.data_test)

            # Pass the split ratio and all keras kwargs down to the member model
            model.train_and_fit(tts=tts, random_state=member_seed, **dict(kwargs))
            self.models.append(model)

            # Attempt to gather scores if the member evaluated on test data
            try:
                score = model.get_scores()
            except Exception:
                score = {"note": "member scores unavailable"}
            self.member_scores.append(score)

            self.progress_bar.update(1)

            # if isinstance(score, dict):
            #     for k, v in score.items():
            #         print(f"  {k}: {v}")

            # Optionally print keras history if available
            # if hasattr(model, "history") and model.history is not None:
            #     hist = getattr(model.history, "history", {})
            #     loss_hist = hist.get("loss", [])
            #     val_loss_hist = hist.get("val_loss", [])
                # if loss_hist:
                #     print(f"  Final train loss: {loss_hist[-1]:.6f}")
                # if val_loss_hist:
                #     print(f"  Final val loss:   {val_loss_hist[-1]:.6f}")

    def _fit(self):
        # Base IModel expects _fit to be called, but we do that manually per-member in train_and_fit
        pass

    def predict(self, x: pd.DataFrame = None) -> np.ndarray:
        if not self.models:
            raise RuntimeError("DeepEnsemble has not been trained yet.")

        # Determine which x to use
        if (self._test_size > 0 or self._pre_split) and x is None:
            x_eval = self._x["test"]
        else:
            x_eval = self._x["full"] if x is None else x

        mus = []
        vars_ = []

        for idx, model in enumerate(self.models):
            if not hasattr(model, "predict_mean_variance"):
                raise RuntimeError(
                    f"Model {idx + 1} ({type(model).__name__}) must implement predict_mean_variance()."
                )

            # Get the predicted mean and variance from the underlying Keras model
            mu_m, var_m = model.predict_mean_variance(x_eval)

            # Removed reshape(-1) to keep the multi-output shape (samples, targets)
            mu_m = np.asarray(mu_m)
            var_m = np.asarray(var_m)

            if mu_m.shape != var_m.shape:
                raise ValueError(
                    f"Member {idx + 1} returned mismatched shapes: {mu_m.shape} vs {var_m.shape}"
                )

            mus.append(mu_m)
            vars_.append(var_m)

        # Use np.stack to create a 3D array: (n_models, n_samples, n_targets)
        mus_stacked = np.stack(mus, axis=0)
        vars_stacked = np.stack(vars_, axis=0)

        self.member_means = mus_stacked
        self.member_vars = vars_stacked

        # Deep Ensemble Logic:
        # np.mean and np.var on axis=0 collapses the models dimension,
        # leaving us with (n_samples, n_targets) arrays.
        self.mean_prediction = np.mean(mus_stacked, axis=0)
        self.epistemic_var = np.var(mus_stacked, axis=0)
        self.aleatoric_var = np.mean(vars_stacked, axis=0)

        self.total_var = self.epistemic_var + self.aleatoric_var
        self.prediction_std = np.sqrt(self.total_var)

        self._predictions = self.mean_prediction

        # Update overall scores if evaluating on the test set
        if (self._test_size > 0 or self._pre_split) and x is None:
            self._score()

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

    def _score(self, folds: int = 5):
        # Call the generic scoring routine from IModel
        super()._score()

        # Update the parameters dictionary with ensemble-specific uncertainty metrics
        self._parameters = {
            "Base Model": self._ensemble_params.get("base_model_class"),
            "Number of Models": self.n_models,
            "Mean Total Std": float(np.mean(self.prediction_std)) if self.prediction_std is not None else None,
            "Mean Epistemic Std": float(
                np.mean(np.sqrt(self.epistemic_var))) if self.epistemic_var is not None else None,
            "Mean Aleatoric Std": float(
                np.mean(np.sqrt(self.aleatoric_var))) if self.aleatoric_var is not None else None,
            "Max Total Std": float(np.max(self.prediction_std)) if self.prediction_std is not None else None,
            "Min Total Std": float(np.min(self.prediction_std)) if self.prediction_std is not None else None,
        }

    def get_parameters(self) -> dict:
        return dict(self._ensemble_params)

    def get_features(self) -> list[str]:
        return self.features

    def get_models(self) -> list[IModel]:
        return self.models

    def print_scores(self):
        print("Deep Ensemble Scores:")
        super().print_scores()

    def print_member_summary(self):
        if not self.member_scores:
            print("No member scores stored.")
            return

        print("\nDeep Ensemble Member Summary:")
        for i, score in enumerate(self.member_scores):
            print(f"\nMember {i + 1}:")
            if isinstance(score, dict):
                for key, value in score.items():
                    print(f"  {key}: {value}")
            else:
                print(f"  {score}")

    def print_training_diagnostics(self):
        if self.member_means is None:
            print("No ensemble predictions computed yet. Run predict(...) first.")
            return

        print("\n Deep Ensemble Diagnostics ")
        print(f"Ensemble size: {len(self.models)}")
        print(f"Mean epistemic variance: {float(np.mean(self.epistemic_var)):.6f}")
        print(f"Mean aleatoric variance: {float(np.mean(self.aleatoric_var)):.6f}")
        print(f"Mean total variance:     {float(np.mean(self.total_var)):.6f}")
        print(
            f"Total std range:         {float(np.min(self.prediction_std)):.6f} "
            f"to {float(np.max(self.prediction_std)):.6f}"
        )

        avg_member_std = np.mean(np.std(self.member_means, axis=0))
        print(f"Average disagreement across members: {float(avg_member_std):.6f}")

        if avg_member_std < 1e-8:
            print("WARNING: ensemble members are nearly identical.")
