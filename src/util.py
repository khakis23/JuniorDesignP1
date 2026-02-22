from datetime import date
from itertools import combinations
import pandas as pd


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


import matplotlib.pyplot as plt
import numpy as np


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


if __name__ == "__main__":
    plot_forecasts({"Test": np.random.randint(0, 7000, 24), "Test": np.random.randint(0, 7000, 24)})