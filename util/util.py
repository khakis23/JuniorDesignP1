from datetime import date
from itertools import combinations
import pandas as pd
import matplotlib.pyplot as plt


def feature_combo(features: list[str], min_list_len=2, max_list_len=5) -> list[list[str]]:
    """
    Get every possible combination of features from a list of features.

    WARNING: This creates a list of O(N^max_list_len) combinations, so be careful with the the number of features!

    :param features:      list of selected features in ModelData
    :param min_list_len:  minimum length of the feature list to generate
    :param max_list_len:  maximum length of the feature list to generate
    :return:  combination of features of length between min_list_len and max_list_len
    """
    res = []

    for r in range(min_list_len, min(len(features) + 1, max_list_len)):
        for combo in combinations(features, r):
            res.append(list(combo))
    return res


def clamp_predictions(predictions, dt_index, elevation_data):
    # Convert raw numpy array to a Series using the provided index
    y_pred = pd.Series(predictions, index=dt_index)

    # Handle duplicates in elevation data
    elev = elevation_data.copy()
    if elev.index.has_duplicates:
        elev = elev[~elev.index.duplicated(keep="last")]

    # Align elevation data to match the prediction timestamps
    elev = elev.reindex(y_pred.index)

    # Create and apply mask
    mask = (elev <= 0).fillna(False).to_numpy()

    # Flatten mask just in case elevation_data is passed as a DataFrame instead of a Series
    if mask.ndim > 1:
        mask = mask.flatten()

    y_pred.iloc[mask] = 0

    # Return the clamped array
    return y_pred.to_numpy()


def plot_forecasts(predictions_dict: dict, date_str=str(date.today()), save_filename=None):
    """
    Plots forecasted PV energy for multiple models on a single graph.

    ** Partially written by AI **

    Args:
        predictions_dict (dict): Keys are model names (str), values are prediction arrays (numpy arrays).
        date_str (str): Date used for the chart title.
        save_filename (str): Optional. If provided, saves the figure to this path.
    """
    hours = np.arange(24)
    fig, ax = plt.subplots(figsize=(10, 5), dpi=200)

    energy_texts = []
    max_energy_text_len = 0

    # Iterate through the dictionary to plot each model
    for model_name, pred_array in predictions_dict.items():
        # Plot the line (Matplotlib automatically cycles colors for you)
        ax.plot(hours, pred_array, linewidth=2.5, label=model_name)

        # Calculate energy and format the text for the box
        energy = pred_array.sum()
        energy_texts.append(f"{model_name}: {energy:,.1f} kWh")

        if len(model_name) > max_energy_text_len:
            max_energy_text_len = len(model_name)

    # Title and Labels
    ax.set_title(f"{date_str} PV Energy Forecast", pad=10)
    ax.set_xlabel("Hour of day")
    ax.set_ylabel("Power (kW)")

    # X-Axis Tick Formatting (00 to 23)
    ax.set_xlim(0, 23)
    ax.set_xticks(hours)
    ax.set_xticklabels([f"{h:02d}" for h in hours])

    # Energy Metrics Box (Combines all strings with a newline)
    energy_display = "\n".join(energy_texts)
    ax.text(
        0.07 + (max_energy_text_len + 11) / 2 * 0.0145, 0.965,
        energy_display,
        transform=ax.transAxes,
        va="top", ha="right",
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#DFF5E1", alpha=0.95, linewidth=0.8),
    )

    # Grid and Spines styling
    ax.grid(True, which="major", linewidth=0.8, alpha=0.35)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Legend and Layout
    ax.legend(frameon=True, loc="upper right")
    fig.tight_layout()

    # Save and Show
    if save_filename:
        plt.savefig(save_filename, dpi=300)
    plt.show()


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def plot_daily_solar_output(y_pred, y_actual=None, title="Daily Solar Output", metrics_dict=None, show_std=True, dpi=300, save_path=None):
    """
    Plots hourly solar profile and Actual vs Predicted scatter.
    If y_actual is None (real-world predictions), it only plots the hourly profile.

    :param y_pred: Predicted solar power values. Must be a pandas Series with a DatetimeIndex to group by hour.
    :param y_actual: Actual solar power values. Can be a pandas Series or DataFrame with a DatetimeIndex. Defaults to None.
    :param title: The main title of the figure. Defaults to "Daily Solar Output".
    :param metrics_dict: Dictionary of text metrics (e.g., {'RMSE': 1.2, 'R²': 0.9}) to display in the top-left box. Defaults to None.
    :param show_std: Boolean flag to display the ±1 standard deviation shaded region on the hourly plot. Defaults to True.
    :param dpi: Resolution of the generated matplotlib figure. Defaults to 100.
    :param save_path: String file path to save the generated figure (e.g., 'plot.png'). If None, figure is only displayed. Defaults to None.
    """
    # Ensure y_pred is a pandas Series with a DatetimeIndex
    if not isinstance(y_pred, pd.Series):
        if y_actual is not None:
            y_pred = pd.Series(y_pred, index=y_actual.index, name="pred")
        else:
            # If no actuals, we can't infer the index. User must pass a Series.
            raise ValueError(
                "For real-world data without actuals, y_pred must be a pandas Series with a DatetimeIndex.")

    has_actual = y_actual is not None
    hours = np.arange(24)

    # predicted — align to hour
    p_mean = y_pred.groupby(y_pred.index.hour).mean().reindex(hours)
    p_std = y_pred.groupby(y_pred.index.hour).std().reindex(hours)
    pred_energy_wh = y_pred.clip(lower=0).sum() * 24 / len(y_pred)

    if has_actual:
        if isinstance(y_actual, pd.DataFrame):
            y_actual = y_actual.iloc[:, 0]

        a_mean = y_actual.groupby(y_actual.index.hour).mean().reindex(hours)
        a_std = y_actual.groupby(y_actual.index.hour).std().reindex(hours)
        actual_energy_wh = y_actual.sum() * 24 / len(y_actual)

        energy_error_pct = (
            100.0 * (pred_energy_wh - actual_energy_wh) / actual_energy_wh
            if actual_energy_wh != 0 else np.nan
        )

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6), dpi=dpi)
    else:
        # Real-world data: only show the hourly profile
        fig, ax1 = plt.subplots(1, 1, figsize=(8, 6), dpi=dpi)
        ax2 = None

    # --- AX1: Hourly Profile ---
    if has_actual:
        ax1.plot(hours, a_mean.values, linewidth=2.5, label="Actual")
    ax1.plot(hours, p_mean.values, linewidth=2.5, label="Predicted")

    if show_std:
        if has_actual and a_std.notna().any():
            ax1.fill_between(hours, (a_mean - a_std).values, (a_mean + a_std).values, alpha=0.15, linewidth=0,
                             label="Actual ±1σ")
        if p_std.notna().any():
            ax1.fill_between(hours, (p_mean - p_std).values, (p_mean + p_std).values, alpha=0.15, linewidth=0,
                             label="Predicted ±1σ")

    ax1.set_title("Average PV Power by Hour", pad=10)
    ax1.set_xlabel("Hour of day")
    ax1.set_ylabel("Power (kW)")
    ax1.set_xlim(0, 23)
    ax1.set_xticks(hours)
    ax1.set_xticklabels([f"{h:02d}" for h in hours])
    ax1.grid(True, which="major", linewidth=0.8, alpha=0.35)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    # Blue metrics box
    if metrics_dict:
        metrics_text = "\n".join([f"{k}: {v}" for k, v in metrics_dict.items()])
        ax1.text(0.02, 0.82, metrics_text, transform=ax1.transAxes, va="top", ha="left", fontsize=10,
                 bbox=dict(boxstyle="round,pad=0.4", facecolor="#DCEBFF", alpha=0.95, linewidth=0.8))

    # Green energy box
    if has_actual:
        energy_metrics = [
            f"Actual Energy: {actual_energy_wh:,.2f} kWh",
            f"Pred Energy: {pred_energy_wh:,.2f} kWh",
            f"Δ Energy: {energy_error_pct:+.2f}%" if np.isfinite(energy_error_pct) else "Δ Energy: N/A",
        ]
    else:
        energy_metrics = [f"Pred Energy: {pred_energy_wh:,.2f} kWh"]

    ax1.text(0.02, 0.96 if metrics_dict else 0.96, "\n".join(energy_metrics), transform=ax1.transAxes, va="top",
             ha="left", fontsize=10,
             bbox=dict(boxstyle="round,pad=0.4", facecolor="#DFF5E1", alpha=0.95, linewidth=0.8))
    ax1.legend(frameon=True, loc="upper right")

    # --- AX2: Actual vs Predicted Scatter (Skip if no actuals) ---
    if ax2 is not None:
        ax2.scatter(y_actual, y_pred, alpha=0.4, edgecolors='none', color='#1f77b4')
        min_val = min(y_actual.min(), min(y_pred))
        max_val = max(y_actual.max(), max(y_pred))

        pad = 50

        ax2.set_xlim(-pad, max(y_actual) + pad)
        ax2.set_ylim(min_val - pad, max_val + pad)

        # Perfect fit line now strictly matches the limits
        ax2.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.75, zorder=3,
                 label="Perfect Fit")

        ax2.set_title("Actual vs Predicted", pad=10)
        ax2.set_xlabel("Actual Power (kW)")
        ax2.set_ylabel("Predicted Power (kW)")
        ax2.grid(True, linestyle=':', alpha=0.6)
        ax2.spines["top"].set_visible(False)
        ax2.spines["right"].set_visible(False)
        ax2.legend(frameon=True, loc="upper left")

    fig.tight_layout()
    fig.suptitle(title, fontsize=18)
    fig.subplots_adjust(top=0.9)  # Give room for suptitle

    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    plot_forecasts({"Test": np.random.randint(0, 7000, 24), "Test": np.random.randint(0, 7000, 24)})
