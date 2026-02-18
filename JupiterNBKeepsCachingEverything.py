from itertools import combinations
from models.RidgeRegression import *
from models.Data import ModelData
import time   # for fun


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


def main1():

    features = [
        "temp",
        "feelslike",
        "dew",
        "humidity",
        "precip",
        "precipprob",
        # "preciptype",   # TODO convert from string to numeric
        "snow",
        "snowdepth",
        "windgust",
        "windspeed",
        "winddir",
        "sealevelpressure",
        "cloudcover",
        "visibility",
        "solarradiation",
        "solarenergy",
        "uvindex",
        "sunazimuth",
        "sunelevation",

        # engineered features
        "hsin",
        "hcos",
        "dsin",
        "dcos"
    ]

    # create a single Data Object to pass into models
    data = ModelData()

    # generate feature combinations
    all_features: list[list[str]] = feature_combo(features, 6, 7)
    all_models: list[RidgeRegression] = []
    ridge_eval: RidgeRegEval

    print(f"Training {len(all_features)} models...")
    start = time.perf_counter()

    # train a model for all combinations of features
    for features in all_features:
        all_models.append(RidgeRegression(features, test_size=0.2, random_state=42))

    # find best model(s)
    print("Evaluating models...")
    ridge_eval = RidgeRegEval(all_models)
    ridge_eval.evaluate()

    # show results
    time_took = time.perf_counter() - start
    print(f"Done! Took {time_took / 60:.2f} minutes — {time_took / len(all_features):.2f} seconds per model.")
    print("\n\tBest model(s):\n")
    ridge_eval.display_best()


def main2():

    features = [
        "temp",
        "cloudcover",
        "uvindex",
        "solarenergy",
        "solarradiation",
        "hsin",
        "hcos",
        "dtemp",
        "dwindspeed",
        "dcloudcover",
        "dsolarradiation",
        "dsolarenergy",
    ]

    # single data object
    data = ModelData()

    # generate feature combinations
    all_features = feature_combo(features, 3, 8)
    all_models = []
    ridge_eval: RidgeRegEval

    print(f"Training {len(all_features)} models...")
    start = time.perf_counter()

    # train and predict with all the models
    for features in all_features:
        all_models.append(RidgeRegression(features, test_size=0.2, random_state=42))

    # find best model(s)
    print("Evaluating models...")
    ridge_eval = RidgeRegEval(all_models)
    ridge_eval.evaluate()

    # show results
    time_took = time.perf_counter() - start
    print(f"Done! Took {time_took / 60:.2f} minutes — {time_took / len(all_features) * 1000:.1f} ms per model.")
    print("\n\tBest model(s):\n")
    ridge_eval.display_best()



if __name__ == "__main__":
    model = RidgeRegression(["cloudcover",  "uvindex",  "solarenergy",  "hcos",  "dtemp",   "dsolarradiation",])
    model.print_results()
    model.plot()

    # main2()
