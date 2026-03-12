import os
import sys
import time
import warnings
from typing import Any

import numpy as np
from sklearn.model_selection import ParameterGrid

from Data.ModelData.ModelData import ModelData
from Models.models.GradientBoostingRegression import GradientBoostingRegression
from Models.models.MLPRegression import MLPRegression
from Models.models.RandomForestRegression import RandomForestRegression
from Models.models.RidgeRegression import RidgeRegression
from Models.models.LSTMRegression import LSTMRegression
from Models.src.ModelEval import ModelEval
from Models.src.IModel import IModel


class Trainer:

    # Implemented models go here
    MODEL_FACTORY = {
        "RidgeRegression": RidgeRegression,
        "LSTMRegression": LSTMRegression,
        "MLPRegression": MLPRegression,
        "RandomForestRegression": RandomForestRegression,
        "GradientBoostingRegression": GradientBoostingRegression,
    }

    def __init__(self):
        self.data = ModelData()
        self.models: list[IModel]

    def test_train(self, model_name: str, features_list: list[list[str]], params: dict[str, list], random_state=42) -> list[IModel]:
        """

        :param model_name:
        :param features_list:
        :param params:
            - NOTE: if being used: train_test_split must be entered as {..., "tts": 0.2, ...}

        :return:
        """
        # reset
        self.models = []

        ### Train, fit, and predict ###
        # create parameter combinations (using this function since already using sklearn)
        param_combos = list(ParameterGrid(params))

        # time tracking
        print(f"Training {len(features_list) * len(param_combos)} models...")
        start_time = time.perf_counter()
        progress_bar = _ProgressBar(len(features_list) * len(param_combos))

        # train models
        for features in features_list:
            for params in param_combos:
                model = self.MODEL_FACTORY[model_name](features, self.data)
                model.train_and_fit(random_state=random_state, **params)
                model.predict()
                self.models.append(model)
                progress_bar.update(1)

        # time tracking
        time_took = time.perf_counter() - start_time
        print(f"\nComplete! Took {time_took / 60:.2f} minutes — {time_took / len(self.models):.3f} s per model.")

        return self.models


def clear_last_line():
    """Moves the cursor up one line and clears it completely."""
    # \033[F moves the cursor to the beginning of the previous line
    # \033[K clears from the cursor to the end of the line
    sys.stdout.write("\033[F")
    sys.stdout.write("\033[K")
    sys.stdout.flush()

class _ProgressBar:
    BLANK = "░"
    FULL = "█"


    def __init__(self, max_steps: int):
        self.total_bars = 30
        self.current_bar = 0
        self.total_steps = 0
        self.ratio = self.total_bars / max_steps
        self._display()

        # keep annoying messages out of the progress bar
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'   # use 2 to also remove warnings
        warnings.filterwarnings("ignore")

    def update(self, update: int) -> None:
        self.total_steps += update
        last = self.current_bar
        self.current_bar = np.ceil(self.total_steps * self.ratio)
        if last != self.current_bar:
            self._display()

    def _display(self):
        bar_chars = [self.FULL if i <= self.current_bar else self.BLANK
                     for i in range(self.total_bars)]
        bar_str = "".join(bar_chars)
        print(f"\r{bar_str}", end="", flush=True)

