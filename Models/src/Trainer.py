import time
from typing import Any

from sklearn.model_selection import ParameterGrid

from Data.ModelData.ModelData import ModelData
from Models.models.GradientBoostingRegression import GradientBoostingRegression
from Models.models.MLPRegression import MLPRegression
from Models.models.RandomForestRegression import RandomForestRegression
from Models.models.RidgeRegression import RidgeRegression
from Models.models.LSTMRegression import LSTMRegression
from Models.src.ModelEval import ModelEval
from Models.src.IModel import IModel
from Models.src.ProgressBar import ProgressBar


class Trainer:

    # Implemented models go here
    MODEL_FACTORY = {
        "RidgeRegression": RidgeRegression,
        "LSTMRegression": LSTMRegression,
        "MLPRegression": MLPRegression,
        "RandomForestRegression": RandomForestRegression,
        "GradientBoostingRegression": GradientBoostingRegression,
        "DeepEnsemble": None,  # TODO!
    }

    def __init__(self):
        self.data = ModelData()
        self.models: list[IModel]
        self.progress_bar = ProgressBar()

    def test_train(self,
                   model_name: str,
                   features_list: list[list[str]],
                   params: dict[str, list],
                   random_search: float=0.0,   # TODO NOT IMPLIMENTED
                   random_state=42
                   ) -> list[IModel]:
        """
        TODO

        :param model_name:
        :param features_list:
        :param params:
            - NOTE: if being used: train_test_split must be entered as {..., "tts": 0.2, ...}
        :param random_search:
        :param random_state:
        :return:
        """
        # reset
        self.models = []

        # ensure all values are lists (deep ensemble params aren't in lists)
        for key, val in params.items():
            if not isinstance(val, list):
                params[key] = [val]
        # create parameter combinations
        param_combos = list(ParameterGrid(params))

        # time tracking
        print(f"Training {len(features_list) * len(param_combos)} models...")
        start_time = time.perf_counter()
        self.progress_bar.set_max_steps(len(features_list) * len(param_combos))

        # train models
        for features in features_list:
            for params in param_combos:
                model = self.MODEL_FACTORY[model_name](features, self.data)
                model.train_and_fit(random_state=random_state, **params)
                model.predict()
                self.models.append(model)
                self.progress_bar.update(1)

        # time tracking
        time_took = time.perf_counter() - start_time
        print(f"\nComplete! Took {time_took / 60:.2f} minutes — {time_took / len(self.models):.3f} s per model.")

        return self.models
