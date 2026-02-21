from abc import ABC, abstractmethod
from models.Data import *

# sklearn
from sklearn.linear_model import *
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score


"""
Use this base class to create new models. Just implement the abstract methods depending on the model.
Implement the IModelEval evaluation method, then use the class to evaluate your models and pick the
best features according to how the evaluation method is set. Call ModelEval.display_best() to see 
the results.
"""
class IModel(ABC):

    def __init__(self, features: list[str], data_obj: ModelData=ModelData(), final=False, **kwargs):
        self.model = None
        self.features = features
        self.elevation_df = data_obj.weather["sunelevation"]

        self._y = data_obj.energy
        self._x = data_obj.features  # includes all features by default
        self._test_size: float

        self.predictions: np.ndarray
        self.y_test: pd.DataFrame
        self.y_train: pd.DataFrame

        self._set_features()
        if final:
            self._final_train_and_fit(**kwargs)
        else:
            self._train_and_fit(**kwargs)
            self._evaluate()

    def predict(self, x: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            raise ValueError("Model not trained yet!")
        self.predictions = self.model.predict(x)
        return self.predictions

    def plot(self, *, title="Average PV Power by Hour", show_std=True, save_path=None, dpi=200):
        """This function is mostly written by GPT 5.2"""
        # --- actual (group to hourly-of-day profile) ---
        yt = self.y_test.copy()
        if isinstance(yt, pd.Series):
            yt = yt.to_frame(name="power")
        y_actual = yt["power"]

        a_mean = y_actual.groupby(y_actual.index.hour).mean()
        a_std = y_actual.groupby(y_actual.index.hour).std()

        # --- predicted (align index to y_test) ---
        y_pred = pd.Series(self.predictions, index=self.y_test.index, name="pred")
        p_mean = y_pred.groupby(y_pred.index.hour).mean()
        p_std = y_pred.groupby(y_pred.index.hour).std()

        hours = np.arange(24)

        # ensure all 0..23 exist (fill missing hours)
        a_mean = a_mean.reindex(hours)
        p_mean = p_mean.reindex(hours)
        a_std = a_std.reindex(hours)
        p_std = p_std.reindex(hours)

        # --- total energy over the full test set (Wh, assuming hourly samples) ---
        actual_energy_wh = y_actual.sum() * 24 / len(y_actual)
        pred_energy_wh = y_pred.clip(lower=0).sum() * 24 / len(y_pred)
        energy_error_pct = (
            100.0 * (pred_energy_wh - actual_energy_wh) / actual_energy_wh
            if actual_energy_wh != 0 else np.nan
        )

        # --- plot ---
        fig, ax = plt.subplots(figsize=(10, 5), dpi=dpi)

        ax.plot(hours, a_mean.values, linewidth=2.5, label="Actual")
        ax.plot(hours, p_mean.values, linewidth=2.5, label="Predicted")

        if show_std:
            if a_std.notna().any():
                ax.fill_between(
                    hours,
                    (a_mean - a_std).values,
                    (a_mean + a_std).values,
                    alpha=0.15,
                    linewidth=0,
                    label="Actual ±1σ",
                )
            if p_std.notna().any():
                ax.fill_between(
                    hours,
                    (p_mean - p_std).values,
                    (p_mean + p_std).values,
                    alpha=0.15,
                    linewidth=0,
                    label="Predicted ±1σ",
                )

        ax.set_title(title, pad=10)
        ax.set_xlabel("Hour of day")
        ax.set_ylabel("Power (W)")

        ax.set_xlim(0, 23)
        ax.set_xticks(hours)
        ax.set_xticklabels([f"{h:02d}" for h in hours])

        ax.grid(True, which="major", linewidth=0.8, alpha=0.35)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        # --- BLUE metrics box (model performance only) ---
        metrics = []
        if hasattr(self, "rmse_clamped"):
            metrics.append(f"RMSE (clamped): {self.rmse_clamped:.2f}")
        if hasattr(self, "r2"):
            metrics.append(f"R²: {self.r2:.3f}")
        if hasattr(self, "ridge") and hasattr(self.ridge, "alpha_"):
            metrics.append(f"α: {self.ridge.alpha_:.3}")

        ax.text(
            0.02, 0.98,
            "\n".join(metrics),
            transform=ax.transAxes,
            va="top", ha="left",
            fontsize=10,
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#DCEBFF", alpha=0.95, linewidth=0.8),
        )

        # --- GREEN energy box (placed below blue box) ---
        energy_metrics = [
            f"Actual Energy: {actual_energy_wh:,.2f} kWh",
            f"Pred Energy: {pred_energy_wh:,.2f} kWh",
            f"Δ Energy: {energy_error_pct:+.2f}%" if np.isfinite(energy_error_pct) else "Δ Energy: N/A",
        ]

        ax.text(
            0.02, 0.80,  # positioned below the first box
            "\n".join(energy_metrics),
            transform=ax.transAxes,
            va="top", ha="left",
            fontsize=10,
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#DFF5E1", alpha=0.95, linewidth=0.8),
        )

        ax.legend(frameon=True, loc="upper right")
        fig.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
        plt.show()

    @abstractmethod
    def print_results(self):
        # this method will be called when ModelEval.display_best() is called
        pass

    # optionally override
    def _set_features(self):
        self._x = self._x[self.features]

    @abstractmethod
    def _train_and_fit(self, **kwargs):
        pass

    @abstractmethod
    def _final_train_and_fit(self, **kwargs):
        pass

    @abstractmethod
    def _evaluate(self):
        # just basic scoring like R2 and RMSE, optionally add clamping evaluation or other methods
        pass


"""
Implement in conjunction with IModel to evaluate many models. display_results() will cleanly
display the best models.
"""
class IModelEval(ABC):

    def __init__(self, models: list[IModel]):
        self.models = models
        self.best_models: list[IModel] = []

    @abstractmethod
    def evaluate(self):
        # compare important features like R2, RMSE, etc., then add to self.best_models
        pass

    def display_best(self):
        for model in self.best_models:
            model.plot()
            print("——— Model Features ——— \n  ", *model.features, sep="  ")
            print("——— Scoring ———\n")
            model.print_results()
