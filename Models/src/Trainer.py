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
        self.progress_bar = _ProgressBar()

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


def clear_last_line():
    """Moves the cursor up one line and clears it completely."""
    # \033[F  moves the cursor to the beginning of the previous line
    # \033[K  clears from the cursor to the end of the line
    sys.stdout.write("\033[F")
    sys.stdout.write("\033[K")
    sys.stdout.flush()


def _warmup_gpu():
    """Triggers the 5070 JIT compilation so it doesn't ruin the progress bar."""
    import tensorflow as tf

    print("Initializing Blackwell Kernels...", end="", flush=True)

    # dummy model to trigger JIT comp warning
    dummy = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(10,)),
        tf.keras.layers.Dense(4, activation='relu'),
        tf.keras.layers.Dense(2, activation='softplus')  # Mean/Var output
    ])
    dummy.compile(optimizer='adam', loss='mse')

    # Run one fake training step to trigger the driver logs
    dummy.fit(np.zeros((1, 10)), np.zeros((1, 2)), epochs=1, verbose=0)
    print(" Done.", flush=True)


class _ProgressBar:
    BLANK = "░"
    FULL = "█"

    def __init__(self, total_bars: int=30, warmup_gpu=True):
        self.total_bars = total_bars
        self.current_bar = 0
        self.total_steps = 0
        self.ratio = 0
        self.max_steps = 0  # for tracking reset and setting ratio

        # keep annoying messages out of the progress bar
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'   # use 2 to also remove warn>
        warnings.filterwarnings("ignore")
        if warmup_gpu:
            _warmup_gpu()

    def set_max_steps(self, max_steps: int) -> None:
        self.max_steps = max_steps
        self.ratio = self.total_bars / max_steps
        self._display()

    def update(self, update: int) -> None:
        if not self.ratio:
            print("\033[38;5;208mMust set max steps before updating!\033[0m")
            return

        # calculate progress bar update
        self.total_steps += update
        last = self.current_bar
        self.current_bar = np.ceil(self.total_steps * self.ratio)

        # correctly display progress bar
        if last != self.current_bar:
            self._display()
        if self.total_steps == self.max_steps:
            self.reset()

    def reset(self):
        self.total_steps = 0
        self.current_bar = 0

    def _display(self):
        bar_chars = [self.FULL if i <= self.current_bar else self.BLANK
                     for i in range(self.total_bars)]
        bar_str = "".join(bar_chars)
        print(f"\r{bar_str}", end="", flush=True)

