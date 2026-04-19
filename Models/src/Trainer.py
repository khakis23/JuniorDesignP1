import os
import random
import sys
import time
import warnings
from typing import Any
import pandas as pd

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
from Models.models.DeepEnsemble import DeepEnsemble
from Models.src.ProgressBar import ProgressBar
from datasets import data_handler as h


class Trainer:

    # Implemented models go here
    MODEL_FACTORY = {
        "RidgeRegression": RidgeRegression,
        "LSTMRegression": LSTMRegression,
        "MLPRegression": MLPRegression,
        "RandomForestRegression": RandomForestRegression,
        "GradientBoostingRegression": GradientBoostingRegression,
        "DeepEnsemble": DeepEnsemble,
    }
    # Built-in datasets go here:
    #   - must return (features, targets, data, <optional>data_test)
    DATABASE_FACTORY = {
        "CHF": h.load_CHF,
        "Reactor": h.load_reactor,
    }

    def __init__(self):
        self.models: list[IModel] = []
        self.progress_bar = ProgressBar()

    def test_train(self,
                   model_name: str,
                   dataset: str | pd.DataFrame,
                   params: dict[str, list],
                   random_state=42,
                   random_search: int=0,
                   features_list: list[list[str]] | None=None,
                   target_cols: list[str] | None=None,
                   plot_func=None,
                   p_bar_enabled=True) -> list[IModel]:
        """

        :param model_name:
        :param features_list:
        :param params:
            - NOTE: if being used: train_test_split must be entered as {..., "tts": 0.2, ...}

        :return:
        """
        self.models = []  # reset

        ### Load dataset ###
        data: pd.DataFrame
        features_list_ = features_list
        data_test = None

        # Load built-in datasets
        if isinstance(dataset, str):
            d = self.DATABASE_FACTORY[dataset]()
            if features_list_ is None:
                features_list_ = [d[0]]
            targets = d[1]
            data = d[2]
            # optional pre-split train/test data
            if len(d) == 4:
                data_test = d[3]

        # Handle custom datasets passed into function
        else:
            targets = target_cols
            data = dataset
            if features_list_ is None:
                raise ValueError("Must provide `features_list` if passing in custom dataset.")

        ### Train, fit, and predict ###
       # convert all values in params to lists for param combinations
        for k, v in params.items():
            if not isinstance(v, list):
                params[k] = [v]

        ## NOTE: hard-coded conversion from str to IModel for base_model_class for DeepEnsembles
        if "base_model_class" in params:
            params["base_model_class"] = [self.MODEL_FACTORY[bm] for bm in params["base_model_class"]]

        # create parameter combinations
        param_combos = list(ParameterGrid(params))

        ## Apply random search if inputted
        if random_search:
            # since: total = params * features = N x N, then sqrt(total) = N
            n = int(np.sqrt(random_search))
            param_combos = random.sample(param_combos, min(n, len(param_combos)))

            # in case there are not enough hyperparameter combinations to fill the square
            remaining = n + n - len(param_combos)
            features_list_ = random.sample(features_list_, min(remaining, len(features_list_)))

        # Time tracking
        n_models = len(features_list_) * len(param_combos)
        print(f"Training {n_models} {model_name} models...")
        start_time = time.perf_counter()

        # Progress bar
        if p_bar_enabled:
            self.progress_bar.set_max_steps(n_models)

        # Train model(s)
        for features in features_list_:
            for p in param_combos:
                model = self.MODEL_FACTORY[model_name](
                    features,
                    targets,
                    data,
                    data_test=data_test,
                    plot_func=plot_func,
                )
                model.train_and_fit(random_state=random_state, **p)
                model.predict()
                self.models.append(model)
                if p_bar_enabled:
                    self.progress_bar.update(1)

        # Time tracking
        time_took = time.perf_counter() - start_time
        print(f"\nComplete! Took {time_took / 60:.2f} minutes — {time_took / len(self.models):.3f} s per model.")
        if p_bar_enabled:
            self.progress_bar.reset()

        return self.models
