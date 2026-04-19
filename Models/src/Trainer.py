from Models.src.ProgressBar import hide_warnings

hide_warnings()  # must go before tf import

import random
import time
import pandas as pd
import numpy as np
from sklearn.model_selection import ParameterGrid
import tensorflow as tf

from Models.src.factories import *
from Models.src.IModel import IModel
from Models.src.ProgressBar import ProgressBar


class Trainer:
    """
        Core execution engine for model initialization, hyperparameter grid searching, and training.

        The Trainer class acts as the bridge between the high-level API (`ModelMaker`) and the
        low-level model implementations (`IModel`). Its primary responsibility is to manage the heavy
        lifting of training loops, data routing, and experimental reproducibility.

        How it Works:
            When provided with a dictionary of hyperparameters,
            it converts them into a comprehensive Cartesian grid. It enforces global random seeds across
            Python, Numpy, and TensorFlow, and then iteratively handles the initialization, training, and prediction
            phases for every model combination in the grid.

        Usage:
            This class is generally not instantiated or interacted with directly by the end-user. Instead,
            it operates entirely under the `ModelMaker` class. `ModelMaker` passes the
            user-defined datasets, feature lists, and parameter grids down to the `Trainer.test_train()`
            method, which executes the computational loops and returns the fully trained models
            back to the evaluator.
        """

    # See factories.py for more info
    MODEL_FACTORY = MODEL_FACTORY
    DATABASE_FACTORY = DATABASE_FACTORY

    def __init__(self):
        self.models: list[IModel] = []
        self.progress_bar = ProgressBar()

    def test_train(self,
                   model_name: str,
                   dataset: str | pd.DataFrame,
                   params: dict[str, list],
                   random_state=None,
                   random_search: int = 0,
                   features_list: list[list[str]] | None = None,
                   target_cols: list[str] | None = None,
                   plot_func=None,
                   p_bar_enabled=True) -> list[IModel]:
        """
        Executes the loop to train models across all feature and hyperparameter grids.

        Responsibilities:
            - Setting random state across Python, Numpy, and TensorFlow.
            - Loading data from pre-defined dataset (see `factories.py`) or a custom passed DataFrame.
            - Instantiated base model if a DeepEnsemble is trained.
                * NOTE: Poorly implemented hard-coded way of doing this...
            - Creating the grid of parameters and features, and applying random search if inputted.
            - Train, fit, and predict for every model combination.

        :param model_name:      String name of the model architecture to train (must exist in MODEL_FACTORY).
        :param dataset:         String referencing a built-in dataset (e.g., "CHF") or a custom pandas DataFrame.
        :param params:          Dictionary of hyperparameters. Values can be single items or lists of items to try.
                                NOTE: `train_test_split` ratio must be passed via this dictionary as `{"tts": [0.2]}`.
        :param random_state:    Integer to lock global seeds across Python, Numpy, and TF, or None for true randomness.
        :param random_search:   Integer representing the target number of models to sample from the total grid space.
        :param features_list:   List containing lists of feature names. Required if passing a custom DataFrame.
                                Example: [["Temp", "Pressure"], ["Temp", "Pressure", "Flow"]]
        :param target_cols:     List of target column names. Required if passing a custom DataFrame.
        :param plot_func:       Optional callable plotting function to bind to the trained models.
        :param p_bar_enabled:   Boolean flag to show or hide the terminal progress bar during the training loop.

        :return:                A list of fully trained `IModel` instances corresponding to every executed combination.
        """
        self.models = []  # reset

        # Set TensorFlow random state
        if random_state is not None:
            random.seed(random_state)
            np.random.seed(random_state)
            tf.random.set_seed(random_state)
            tf.keras.utils.set_random_seed(random_state)
            tf.config.experimental.enable_op_determinism()

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

        # Train model(s) in grid search
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