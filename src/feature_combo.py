from itertools import combinations


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
