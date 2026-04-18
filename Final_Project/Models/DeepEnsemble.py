import numpy as np
from Models.src.IModel import IModel


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

    def __init__(
        self,
        features: list[str],
        data,
        base_model_class=None,
        n_models: int = 5,
    ):
        super().__init__(features, data) 
        self.base_model_class = base_model_class # Building ensemble members
        self.n_models = n_models # Number of models in the ensemble

        # Stores trained models and their scores 
        self.models: list[IModel] = []
        self.member_scores = []

        # Stores predicited values
        self.mean_prediction = None # Final avg perdiction 
        self.prediction_std = None # std of predicitons 
        self.epistemic_var = None # Variance from disagreement 
        self.aleatoric_var = None # Variance from noise 
        self.total_var = None # Total predictive variance 

        # Stores predicitions from models
        self.member_means = None
        self.member_vars = None

        self._ensemble_params = {} 

    def train_and_fit(self, random_state=42, **kwargs):
        tts = kwargs.pop("tts", 0.0)
        super().train_and_fit(tts, random_state)

        # Get model type and number of ensemble members
        base_model_class = kwargs.pop("base_model_class", self.base_model_class)
        n_models = kwargs.pop("n_models", self.n_models)

        if base_model_class is None:
            raise ValueError("DeepEnsemble requires `base_model_class`.")

        self.base_model_class = base_model_class
        self.n_models = n_models
        self.models = [] # Resets models
        self.member_scores = [] # Resetd scores

        # Parameters
        self._ensemble_params = {
            "base_model_class": getattr(base_model_class, "__name__", str(base_model_class)),
            "n_models": n_models,
            **kwargs,
        }

        print(f"\nTraining Deep Ensemble with {n_models} members...")

        for i in range(n_models):
            member_seed = None if random_state is None else random_state + i

            print(f"\n--- Training member {i + 1}/{n_models} ---")
            model = base_model_class(self.features, self.data) # Creates model instance
            model.train_and_fit(random_state=member_seed, tts=tts, **dict(kwargs)) # Train model
            self.models.append(model) # Stores trained model

            # Collects model performance
            try:
                score = model.get_scores()
            except Exception:
                score = {"note": "member scores unavailable"}
            self.member_scores.append(score)

            if isinstance(score, dict):
                for k, v in score.items():
                    print(f"  {k}: {v}")

            if hasattr(model, "history") and model.history is not None:
                hist = getattr(model.history, "history", {})
                loss_hist = hist.get("loss", [])
                val_loss_hist = hist.get("val_loss", [])
                if loss_hist:
                    print(f"  Final train loss: {loss_hist[-1]:.6f}")
                if val_loss_hist:
                    print(f"  Final val loss:   {val_loss_hist[-1]:.6f}")

        self._fit()

    def predict(self, X):
        # Makes sure that models are trained
        if not self.models:
            raise RuntimeError("DeepEnsemble has not been trained yet.")

        mus = [] # Prediction values (means)
        vars_ = [] # Prediction uncertainy (variance)

        for idx, model in enumerate(self.models):
            # Ensure model supports uncertainity 
            if not hasattr(model, "predict_mean_variance"):
                raise RuntimeError(
                    f"Model {idx + 1} ({type(model).__name__}) must implement predict_mean_variance()."
                )

            # Get prediciton mean and variance
            mu_m, var_m = model.predict_mean_variance(X)
            mu_m = np.asarray(mu_m).reshape(-1)
            var_m = np.asarray(var_m).reshape(-1)

            # Check that sizes match
            if mu_m.shape != var_m.shape:
                raise ValueError(
                    f"Member {idx + 1} returned mismatched shapes: {mu_m.shape} vs {var_m.shape}"
                )

            mus.append(mu_m)
            vars_.append(var_m)

        # Makes it in matrices
        mus = np.vstack(mus) # shape (M, N)
        vars_ = np.vstack(vars_) # shape (M, N)

        #Stores member predicitons
        self.member_means = mus
        self.member_vars = vars_

        # Computes ensemble
        self.mean_prediction = np.mean(mus, axis=0) # Avg prediction 
        self.epistemic_var = np.var(mus, axis=0, ddof=0) # Model disagreement
        self.aleatoric_var = np.mean(vars_, axis=0) # Data noise
        self.total_var = np.maximum(self.epistemic_var + self.aleatoric_var, 0.0)
        self.prediction_std = np.sqrt(self.total_var) # convert variance to std

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

    def _score(self):
        super()._score()

        self._parameters = {
            "Base Model": self._ensemble_params.get("base_model_class"),
            "Number of Models": self.n_models,
            "Mean Total Std": float(np.mean(self.prediction_std)) if self.prediction_std is not None else None,
            "Mean Epistemic Std": float(np.mean(np.sqrt(self.epistemic_var))) if self.epistemic_var is not None else None,
            "Mean Aleatoric Std": float(np.mean(np.sqrt(self.aleatoric_var))) if self.aleatoric_var is not None else None,
            "Max Total Std": float(np.max(self.prediction_std)) if self.prediction_std is not None else None,
            "Min Total Std": float(np.min(self.prediction_std)) if self.prediction_std is not None else None,
        }

    def get_parameters(self) -> dict:
        return {**self._ensemble_params, **self._parameters}

    def get_features(self) -> list[str]:
        return self.features

    def get_models(self) -> list[IModel]:
        return self.models

    def print_scores(self):
        print("Deep Ensemble Scores:")
        for key, value in self._scores.items():
            print(f"  {key}: {value}")

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

        print("\n------------ Deep Ensemble Diagnostics ------------")
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

        print("-------------------------------------------------------")
        
    def get_plot_arrays(self, X, y_true=None, sort_by_index: bool = False):
        """
        Runs prediction and returns all arrays needed for plotting.
        """

        # Run prediciton and get uncertianity outputs 
        preds = self.predict_with_uncertainty(X)
        
        # Coverts output into 1D for ploting 
        ensemble_mean = np.asarray(preds["mean"]).reshape(-1)
        ensemble_std = np.asarray(preds["std"]).reshape(-1)
        total_var = np.asarray(preds["total_var"]).reshape(-1)
        epistemic_var = np.asarray(preds["epistemic_var"]).reshape(-1)
        aleatoric_var = np.asarray(preds["aleatoric_var"]).reshape(-1)
        member_means = np.asarray(preds["member_means"])
        member_vars = np.asarray(preds["member_vars"])
        
        # Number of prediction points 
        n_samples = len(ensemble_mean)

        # Orders points on the x-axis
        if sort_by_index:
            sort_idx = np.arange(n_samples)
            x_axis = np.arange(n_samples)
            x_label = "Sample Index"
        else:
            sort_idx = np.argsort(ensemble_mean)
            x_axis = np.arange(n_samples)
            x_label = "Sample (sorted by ensemble mean)"

        # Building functions for arrays to be plotted
        plot_data = {
            "x_axis": x_axis,
            "x_label": x_label,
            "sort_idx": sort_idx,
            "ensemble_mean": ensemble_mean[sort_idx],
            "ensemble_std": ensemble_std[sort_idx],
            "total_var": total_var[sort_idx],
            "epistemic_var": epistemic_var[sort_idx],
            "aleatoric_var": aleatoric_var[sort_idx],
            "member_means": member_means[:, sort_idx],
            "member_vars": member_vars[:, sort_idx],
        }
        
        # Add actual outputs if provided
        if y_true is not None:
            y_true = np.asarray(y_true).reshape(-1)
            plot_data["y_true"] = y_true[sort_idx]
        else:
            plot_data["y_true"] = None

        return plot_data
