from Models.src.ProgressBar import hide_warnings
hide_warnings()  # must go before tf import

import tensorflow as tf
import pandas as pd

from Models.src.IModel import IModel
from Models.src.Trainer import Trainer
from Models.src.ModelEval import ModelEval


class ModelMaker:
    """
    API wrapper class for model training, evaluation, and comparison.

    ModelMaker abstracts the underlying model training and evaluation logic and allows for easy comparison of
    unique combinations of parameters and models.

    Workflow:
        1. train_and_eval() — gridsearch training will train and evaluate all combinations of parameters.
        2. choose_best() — train_and_eval() will display a best model for each type of score.
            2a. Directly access model's methods (using ModelMaker.best.<method>) for further analysis.
            2b. Retrain the best model on all data, and make predictions.
    """

    def __init__(self):
        self.trainer = Trainer()
        self.eval: ModelEval = ModelEval()
        self.best: IModel = None
        self.best_models = []
        self.cur_model_name: str = "NULL"
        self._saved_models = []

        devices = tf.config.list_physical_devices()
        print(f"Available Processors: {devices}")  # if GPU appears, then it will be used

    def train_and_eval(self,
                       model_name: str,
                       dataset: pd.DataFrame | str,
                       features_list: list[list[str]],
                       params: dict[str, list],
                       target_cols: list[str] = None,
                       plot_func=None,
                       random_state=42,
                       random_search: int=0,
                       display_best=True,
                       autosave: str="",
                       ):
        """
        Train and evaluate a combination of a chosen model.

        :param model_name:      String name from available models (see Trainer for list of models)
        :param features_list:   List of a list of features from ModelData
                                EXAMPLE:
                                    [["temp", "humidity", "wind_speed"], [...], ...]
        :param params:          Parameters for the chosen model, with keys matching model's parameter kwargs, and
                                values as a list of parameters to try.
                                EXAMPLE:
                                    `{"alpha": [0.1, 0.2, 0.3], "hidden_layer_sizes": [(100, 100), (200, 200)]}`
                                NOTES:
                                    - If being used: train_test_split must be entered as {..., "tts": [0.2], ...}
                                    - Single value parameters must be wrapped in a list (e.g.  {"tts": 0.2} becomes {"tts": [0.2]})
        :param random_state:    Random state for train_test_split and confidence interval
        :param random_search:   TODO NOT IMPLEMENTED
        :param display_best:    Boolean flag to display the best model's score and parameters (and plot(s)
                                    if IModel method overridden)
        :param autosave:        String name of the ModelEval score key  (examples: "R2", "MAE", "RMSE", "CI")
        """
        # reset
        self.cur_model_name = model_name
        self.eval.clear_models()

        # train and evaluate
        self.eval.add_models(self.trainer.test_train(
            model_name,
            dataset,
            params,
            random_search=random_search,
            target_cols=target_cols,
            features_list=features_list,
            plot_func=plot_func,
            random_state=random_state))
        self.best_models = self.eval.evaluate()

        # display best models
        if display_best:
            self.eval.display_best()

        # automatically select the best model with score "autosave"
        if autosave:
            self.best = self.eval.best_models.get(autosave)
            if not self.best:
                raise ValueError(f"ModelEval did not find score \"{autosave}\" in best_models.")

            # save and display model
            print(f"The following {self.cur_model_name} has been saved at index {len(self._saved_models)}...")
            self._saved_models.append(self.best)
            self.best.print_parameters()

    def train_eval_deep_ens(self,
                            dataset: pd.DataFrame | str,
                            best_idx: int = None,
                            base_model: str = None,
                            target_cols: list[str] | None = None,
                            n_models: int = 5,
                            features: list[list[str]] = None,
                            params: dict = {},
                            plot_func=None,
                            random_state=42,
                            auto_save: str | bool = False,
                            ):
        """

            - If best_idx is None and base_model is None, then self.best is used.

        :param dataset:
        :param best_idx:
        :param target_cols:
        :param base_model:
        :param n_models:
        :param features:
        :param params:
        :param random_state:
        :return:
        """
        ### Figure out how model params were passed into function
        model: IModel | None = None
        params_ = params

        # Model by saved index
        if best_idx:
            model = self.get_saved_model(best_idx)

        # Model inferred from autosaved or selected best model
        elif not base_model:
            if self.best:
                model = self.best
            else:
                raise ValueError("ModelMaker.train_deep_ensemble() requires a selected best model if no base_model is provided.")

        # Reused a previously trained model's parameters
        if model:
            # pull hyperparams and features from model
            params_ = model.get_parameters()
            features = [model.get_features()]
            target_cols = model.get_targets()

            # add model name to params
            params_["base_model_class"] = model.__class__.__name__

        # Incorrect input combination: there are 3 different ways to pass parameters for this method (see docstring)
        elif not base_model or not features or not target_cols:
            raise ValueError("ModelMaker.train_deep_ensemble() either requires a base_model, features, "
                             "and target_cols, or a selected best model.")

        # Manually inputted parameters (correctly)
        else:
            params_["base_model_class"] = base_model

        if n_models:  # TODO other paramerter overides when selecting existing model??
            params_["n_models"] = n_models

        ### Train Deep Ensemble
        deep_ens_s = self.trainer.test_train(
            model_name="DeepEnsemble",
            dataset=dataset,
            params=params_,
            features_list=features,
            target_cols=target_cols,
            plot_func=plot_func,
            random_state=random_state,
            p_bar_enabled=False,
        )

        ### Evaluate
        self.eval.add_models(deep_ens_s)

        # multiple models
        if len(deep_ens_s) > 1:
            self.best_models = self.eval.evaluate()
            self.eval.display_best()

        # single model
        else:
            self.best_models = self.eval.display_best(deep_ens_s[0])

        # autosave
        if auto_save:
            if isinstance(auto_save, str):
                self.save_best(self.eval.best_models.get(auto_save))
            else:
                self.save_best(0)

    def save_best(self, idx: int) -> IModel:
        """
        Set the best model to self.best, which give public access to the best model object.

        :param idx: index of the models that have displayed using train_and_eval()

        Example usage after choosing best model, directly access model's methods::

            ModelMaker.best.plot()
            scores: dict = ModelMaker.best.get_scores()

        """
        if not self.best_models:
            raise IndexError("ModelMaker.choose_best() cannot be called before train_and_eval().")
        self.best = self.best_models[idx]

        # save the model for later
        print(f"Saved {self.cur_model_name} at index {len(self._saved_models)}.")
        self._saved_models.append(self.best)

        return self.best

    def clear_evaluator(self):
        self.eval.clear_all()

    def get_saved_model(self, idx: int) -> IModel | None:
        try:
            return self._saved_models[idx]
        except IndexError as e:
            print("get_saved_model() index out of range!\n", e)
            return None
