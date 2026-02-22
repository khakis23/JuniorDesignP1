from abc import ABC, abstractmethod
from models.Data import *

# sklearn
from sklearn.linear_model import *
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.neural_network import MLPRegressor


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

    def plot(self, show_std=True, save_path=None, dpi=200):
        # actual — group to hourly-of-day profile
        yt = self.y_test.copy()
        if isinstance(yt, pd.Series):
            yt = yt.to_frame(name="power")
        y_actual = yt["power"]

        a_mean = y_actual.groupby(y_actual.index.hour).mean()
        a_std = y_actual.groupby(y_actual.index.hour).std()

        # predicted — align index to y_test
        y_pred = pd.Series(self.predictions, index=self.y_test.index, name="pred")
        p_mean = y_pred.groupby(y_pred.index.hour).mean()
        p_std = y_pred.groupby(y_pred.index.hour).std()

        hours = np.arange(24)

        # ensure all 0..23 exist (fill missing hours)
        a_mean = a_mean.reindex(hours)
        p_mean = p_mean.reindex(hours)
        a_std = a_std.reindex(hours)
        p_std = p_std.reindex(hours)

        # total energy over the full test set
        actual_energy_wh = y_actual.sum() * 24 / len(y_actual)
        pred_energy_wh = y_pred.clip(lower=0).sum() * 24 / len(y_pred)
        energy_error_pct = (
            100.0 * (pred_energy_wh - actual_energy_wh) / actual_energy_wh
            if actual_energy_wh != 0 else np.nan
        )

        # plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6), dpi=dpi)

        # hourly profile
        ax1.plot(hours, a_mean.values, linewidth=2.5, label="Actual")
        ax1.plot(hours, p_mean.values, linewidth=2.5, label="Predicted")

        if show_std:
            if a_std.notna().any():
                ax1.fill_between(
                    hours,
                    (a_mean - a_std).values,
                    (a_mean + a_std).values,
                    alpha=0.15,
                    linewidth=0,
                    label="Actual ±1σ",
                )
            if p_std.notna().any():
                ax1.fill_between(
                    hours,
                    (p_mean - p_std).values,
                    (p_mean + p_std).values,
                    alpha=0.15,
                    linewidth=0,
                    label="Predicted ±1σ",
                )

        ax1.set_title("Average PV Power by Hour", pad=10)
        ax1.set_xlabel("Hour of day")
        ax1.set_ylabel("Power (kW)")

        ax1.set_xlim(0, 23)
        ax1.set_xticks(hours)
        ax1.set_xticklabels([f"{h:02d}" for h in hours])

        ax1.grid(True, which="major", linewidth=0.8, alpha=0.35)
        ax1.spines["top"].set_visible(False)
        ax1.spines["right"].set_visible(False)

        # blue metrics box
        metrics = []
        if hasattr(self, "rmse_clamped"):
            metrics.append(f"RMSE (clamped): {self.rmse_clamped:.2f}")
        if hasattr(self, "r2"):
            metrics.append(f"R²: {self.r2:.3f}")
        if hasattr(self, "ridge") and hasattr(self.ridge, "alpha_"):
            metrics.append(f"α: {self.ridge.alpha_:.3e}")

        if metrics:
            ax1.text(
                0.02, 0.96,
                "\n".join(metrics),
                transform=ax1.transAxes,
                va="top", ha="left",
                fontsize=10,
                bbox=dict(boxstyle="round,pad=0.4", facecolor="#DCEBFF", alpha=0.95, linewidth=0.8),
            )

        # energy box
        energy_metrics = [
            f"Actual Energy: {actual_energy_wh:,.2f} kWh",
            f"Pred Energy: {pred_energy_wh:,.2f} kWh",
            f"Δ Energy: {energy_error_pct:+.2f}%" if np.isfinite(energy_error_pct) else "Δ Energy: N/A",
        ]

        ax1.text(
            0.02, 0.82,
            "\n".join(energy_metrics),
            transform=ax1.transAxes,
            va="top", ha="left",
            fontsize=10,
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#DFF5E1", alpha=0.95, linewidth=0.8),
        )
        ax1.legend(frameon=True, loc="upper right")

        # ax2 — actual vs predicted scatter plot
        ax2.scatter(y_actual, self.predictions, alpha=0.4, edgecolors='none', color='#1f77b4')

        # Calculate limits to make the diagonal line perfect
        min_val = min(y_actual.min(), min(self.predictions))
        max_val = max(y_actual.max(), max(self.predictions))
        pad = (max_val - min_val) * 0.05

        ax2.set_xlim(min_val - pad, max_val + pad)
        ax2.set_ylim(min_val - pad, max_val + pad)

        # Perfect Fit Line
        ax2.plot([min_val - pad, max_val + pad], [min_val - pad, max_val + pad], 'r--', alpha=0.75, zorder=3,
                 label="Perfect Fit")

        ax2.set_title("Actual vs Predicted", pad=10)
        ax2.set_xlabel("Actual Power (kW)")
        ax2.set_ylabel("Predicted Power (kW)")
        ax2.grid(True, linestyle=':', alpha=0.6)

        ax2.spines["top"].set_visible(False)
        ax2.spines["right"].set_visible(False)
        ax2.legend(frameon=True, loc="upper left")

        fig.tight_layout()
        fig.suptitle(self.__class__.__name__, fontsize=18)

        if save_path:
            fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
        plt.show()

    def _clamp_predictions(self):
        """
        Alter self.predictions to remove impossible output values based on elevation data.
        """
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

    # optionally override
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
