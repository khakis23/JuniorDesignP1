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
