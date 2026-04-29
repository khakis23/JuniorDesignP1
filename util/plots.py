from datetime import date
from Models.models.DeepEnsemble import DeepEnsemble
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from util.SavePaths import SavePaths

import matplotlib
if SavePaths.save_path is not None:
    matplotlib.use('Agg')   # For leadless server

"""
NOTE: The functions in this file are mostly AI generated, then human-verified. 
      Since these functions have access to the model's public members, it allows
      the plots to access all the important data from an trained model.
"""


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


def plot_de(ensemble: DeepEnsemble, X_test: np.ndarray, y_true: np.ndarray = None,
            figsize: tuple = (14, 6), alpha: float = 0.3, sort_by_index: bool = False):
    """
    DEPRECATED!
    """

    # Get all plot-ready arrays from the ensemble
    plot_data = ensemble.get_plot_arrays(X_test, y_true=y_true, sort_by_index=sort_by_index)

    x_axis = plot_data["x_axis"]
    x_label = plot_data["x_label"]
    member_means = plot_data["member_means"]
    ensemble_mean = plot_data["ensemble_mean"]
    ensemble_std = plot_data["ensemble_std"]
    y_true_sorted = plot_data["y_true"]

    n_models = member_means.shape[0]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    colors = plt.colormaps['tab10'](np.linspace(0, 1, n_models))

    # Plot 1: member predictions
    for i in range(n_models):
        ax1.plot_func(
            x_axis,
            member_means[i, :],
            alpha=alpha,
            linewidth=0.8,
            color=colors[i],
            label=f"Member {i + 1}"
        )

    if y_true_sorted is not None:
        ax1.scatter(
            x_axis[::max(1, len(x_axis) // 100)],
            y_true_sorted[::max(1, len(x_axis) // 100)],
            alpha=0.4,
            s=10,
            color="black",
            label="True values",
            zorder=10
        )

    ax1.set_xlabel(x_label, fontsize=12)
    ax1.set_ylabel("Prediction", fontsize=12)
    ax1.set_title(f"Ensemble Members (n={n_models})", fontsize=14, fontweight="bold")
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="best", fontsize=9)

    # Plot 2: ensemble mean with uncertainty
    ax2.fill_between(
        x_axis,
        ensemble_mean - ensemble_std,
        ensemble_mean + ensemble_std,
        alpha=0.3,
        color="blue",
        label="±1 std (total)"
    )

    ax2.fill_between(
        x_axis,
        ensemble_mean - 2 * ensemble_std,
        ensemble_mean + 2 * ensemble_std,
        alpha=0.15,
        color="blue",
        label="±2 std (total)"
    )

    ax2.plot_func(
        x_axis,
        ensemble_mean,
        "b-",
        linewidth=2,
        label="Ensemble mean",
        zorder=5
    )

    if y_true_sorted is not None:
        ax2.scatter(
            x_axis[::max(1, len(x_axis) // 100)],
            y_true_sorted[::max(1, len(x_axis) // 100)],
            alpha=0.4,
            s=10,
            color="red",
            label="True values",
            zorder=10
        )

    ax2.set_xlabel(x_label, fontsize=12)
    ax2.set_ylabel("Prediction", fontsize=12)
    ax2.set_title("Ensemble Mean with Uncertainty Bands", fontsize=14, fontweight="bold")
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc="best", fontsize=9)

    plt.tight_layout()
    plt.show()

    return fig, (ax1, ax2)


def plot_deep_ensemble_eval(model, x_dict, y_dict, figsize=(16, 6), title_suffix="", **kwargs):
    # determine which split to plot
    is_split = model._test_size > 0 or model._pre_split
    x_eval = x_dict["test"] if is_split else x_dict["full"]
    y_eval = y_dict["test"] if is_split else y_dict["full"]

    # generate predictions
    model.predict(x_eval)

    y_eval_arr = np.asarray(y_eval)

    # Extract only the first target if dataset is multi-target
    if y_eval_arr.ndim > 1 and y_eval_arr.shape[1] > 1:
        mean_pred = model.mean_prediction[:, 0]
        std_pred = model.prediction_std[:, 0]
        y_true = y_eval_arr[:, 0]
    else:
        mean_pred = model.mean_prediction.ravel()
        std_pred = model.prediction_std.ravel()
        y_true = y_eval_arr.ravel()

    member_means = model.member_means

    # sort data by true values for a readable curve
    sort_idx = np.argsort(y_true)
    y_sorted = y_true[sort_idx]
    x_axis = np.arange(len(y_sorted))

    # ==========================================
    # DYNAMIC TARGET HANDLING
    # ==========================================
    if hasattr(model, 'targets') and model.targets:
        target_name = ", ".join(model.targets)
    else:
        target_name = "Target"

    y_axis_label = f"Target Value ({target_name})"

    # ==========================================
    # DYNAMIC SCALING FOR LARGE ENSEMBLES
    # ==========================================
    n = model.n_models
    line_alpha = min(1.0, max(0.05, 1.5 / np.sqrt(n)))
    line_width = max(0.5, 2.0 / np.sqrt(n))
    group_members = n > 10

    fig, axes = plt.subplots(1, 2, figsize=figsize)
    colors = plt.colormaps['tab10'](np.linspace(0, 1, min(n, 10)))

    # ==========================================
    # PLOT 1: Individual Member Predictions
    # ==========================================
    ax1 = axes[0]
    ax1.scatter(x_axis, y_sorted, color='black', alpha=0.3, label="True Data", s=10, zorder=3)

    for i in range(n):
        member_pred_sorted = member_means[i, sort_idx, 0]

        if group_members:
            color = '#17becf'
            label = "Ensemble Members" if i == 0 else None
        else:
            color = colors[i]
            label = f"Member {i + 1}"

        ax1.plot(x_axis, member_pred_sorted, color=color, alpha=line_alpha,
                 linewidth=line_width, label=label, zorder=2)

    ax1.set_title(f"Individual Model Predictions {title_suffix}", fontsize=16, pad=10)
    ax1.set_xlabel("Sample Index (Sorted by True Value)", fontsize=12)
    ax1.set_ylabel(y_axis_label, fontsize=12)
    ax1.set_xlim(0, len(y_sorted) - 1)

    ax1.legend(loc="upper left", frameon=True, fontsize=10)
    ax1.grid(True, linestyle=':', alpha=0.6)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    # ==========================================
    # PLOT 2: Ensemble Mean & Variance
    # ==========================================
    ax2 = axes[1]
    ax2.scatter(x_axis, y_sorted, color='black', alpha=0.3, label="True Data", s=10, zorder=3)

    mean_sorted = mean_pred[sort_idx]
    std_sorted = std_pred[sort_idx]

    ax2.plot(x_axis, mean_sorted, color='blue', linewidth=1.5, label="Ensemble Mean", zorder=4)

    ax2.fill_between(x_axis,
                     mean_sorted - std_sorted,
                     mean_sorted + std_sorted,
                     color='blue', alpha=0.3, label="±1 Std Dev", zorder=1)

    ax2.fill_between(x_axis,
                     mean_sorted - 2 * std_sorted,
                     mean_sorted + 2 * std_sorted,
                     color='blue', alpha=0.15, label="±2 Std Dev (95% CI)", zorder=0)

    ax2.set_title(f"Ensemble Uncertainty {title_suffix}", fontsize=16, pad=10)
    ax2.set_xlabel("Sample Index (Sorted by True Value)", fontsize=12)
    ax2.set_ylabel(y_axis_label, fontsize=12)
    ax2.set_xlim(0, len(y_sorted) - 1)

    ax2.legend(loc="upper left", frameon=True, fontsize=10)
    ax2.grid(True, linestyle=':', alpha=0.6)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    # ==========================================
    # RUBRIC REQUIREMENT: DECOMPOSITION METRICS
    # ==========================================
    scores = model.get_scores()
    r2 = 0
    if scores:
        r2 = scores.get("R2", np.nan)
        rmse = scores.get("RMSE", np.nan)

        # Retrieve pre-calculated metrics directly from the dictionary keys
        m_total_std = scores.get("totalSTD", np.nan)
        m_epi_std = scores.get("epSTD", np.nan)
        m_ale_std = scores.get("alSTD", np.nan)
        coverage = scores.get("95 CI", np.nan)

        text_str = (f"Model Metrics:\n"
                    f"$R^2$: {r2:.3f}\n"
                    f"RMSE: {rmse:.1f}\n"
                    f"Total $\sigma$: {m_total_std:.1f}\n"
                    f"Epistemic $\sigma$: {m_epi_std:.1f}\n"
                    f"Aleatoric $\sigma$: {m_ale_std:.1f}\n"
                    f"95% CI: {coverage:.3f}")

        props = dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.85, edgecolor='#cccccc')
        ax2.text(0.96, 0.96, text_str, transform=ax2.transAxes, fontsize=11,
                 verticalalignment='top', horizontalalignment='right', bbox=props, zorder=5)

    plt.tight_layout()
    if SavePaths.save_path is not None:
        plt.savefig(SavePaths.save_path / f"DE_r2-{round(r2, 8)}.png", dpi=300, bbox_inches='tight')
    else:
        plt.show()


def plot_actual_vs_pred(model, x_dict, y_dict, figsize=(10, 8), title="Actual vs Predicted", **kwargs):
    # select the correct split
    is_split = model._test_size > 0 or model._pre_split
    x_eval = x_dict["test"] if is_split else x_dict["full"]
    y_eval = y_dict["test"] if is_split else y_dict["full"]

    # generate predictions (returns mean only)
    y_pred = model.predict(x_eval).ravel()
    y_actual = np.asarray(y_eval).ravel()

    fig, ax = plt.subplots(figsize=figsize)

    min_val = min(y_actual.min(), y_pred.min())
    max_val = max(y_actual.max(), y_pred.max())
    pad = (max_val - min_val) * 0.1
    limit_range = [min_val - pad, max_val + pad]

    ax.scatter(y_actual, y_pred, s=40, alpha=0.4, edgecolors='none', color='#1f77b4', label="Predictions")
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.75, zorder=3, label="Perfect Fit")

    ax.set_xlim(limit_range)
    ax.set_ylim(limit_range)

    target_name = model.targets[0] if model.targets else "Target"
    ax.set_xlabel(f"Actual {target_name}")
    ax.set_ylabel(f"Predicted {target_name}")

    ax.set_title(title, pad=15, fontsize=14)
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=True, loc="upper left")

    plt.tight_layout()
    if SavePaths.save_path is not None:
        r2 = model.get_scores().get("R2", 0)
        plt.savefig(SavePaths.save_path / f"act_vs_pred_{model.__class__.__name__}_r2-{round(r2, 8)}.png", dpi=300, bbox_inches='tight')
    else:
        plt.show()


def plot_parity_with_uncertainty(model, x_dict, y_dict, figsize=(10, 8), title="parity plot with 95% ci", **kwargs):
    is_split = model._test_size > 0 or model._pre_split
    x_eval = x_dict["test"] if is_split else x_dict["full"]
    y_eval = y_dict["test"] if is_split else y_dict["full"]

    y_pred = model.predict(x_eval).ravel()
    y_actual = np.asarray(y_eval).ravel()

    # Smart standard deviation extraction to prevent single MLP crashes
    if hasattr(model, 'prediction_std') and model.prediction_std is not None:
        std = model.prediction_std.ravel()
    elif hasattr(model, 'predict_mean_variance'):
        _, var = model.predict_mean_variance(x_eval)
        std = np.sqrt(var).ravel()
    else:
        std = np.zeros_like(y_pred)

    y_err = 1.96 * std

    fig, ax = plt.subplots(figsize=figsize)

    min_val = min(y_actual.min(), y_pred.min())
    max_val = max(y_actual.max(), y_pred.max())
    pad = (max_val - min_val) * 0.1
    limit_range = [min_val - pad, max_val + pad]

    ax.errorbar(y_actual, y_pred, yerr=y_err, fmt='none', ecolor='#1f77b4', alpha=0.2, label="95% CI")
    ax.scatter(y_actual, y_pred, s=40, alpha=0.6, edgecolors='white', color='#1f77b4', label="mean prediction")
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.75, zorder=3, label="perfect fit")

    ax.set_xlim(limit_range)
    ax.set_ylim(limit_range)

    target_name = model.targets[0] if model.targets else "target"
    ax.set_xlabel(f"actual {target_name}")
    ax.set_ylabel(f"predicted {target_name}")
    ax.set_title(title, pad=15, fontsize=14)
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=True, loc="upper left")

    plt.tight_layout()
    # plt.show()
    if SavePaths.save_path is not None:
        r2 = model.get_scores().get("R2", 0)
        plt.savefig(SavePaths.save_path / f"par_uncert_{model.__class__.__name__}_r2-{round(r2, 8)}.png", dpi=300, bbox_inches='tight')
    else:
        plt.show()


def deep_ensemble_wrapper(model, x, y, **kwargs):
    plot_parity_with_uncertainty(model, x, y, **kwargs)
    plot_deep_ensemble_eval(model, x, y, **kwargs)
