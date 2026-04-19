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

    ModelMaker abstracts the underlying model training and evaluation logic, allowing users
    to easily execute grid searches, compare unique combinations of hyperparameters, and
    generate deep ensembles without interacting directly with the lower-level architecture.

    General Class Workflow:
        1. Instantiate `ModelMaker()`.
        2. Define data (pd.DataFrame or built-in string), features, and target columns.
            - See `factories.py` for built-in datasets.
        3. Define a dictionary of parameters for the target model using the syntax of the specific model's implementation.
        4. Execute `train_and_eval()` to run a grid or random search. This displays the best models for each metric.
        5. (Optional) Call `save_best()` to capture a specific model from the grid search into memory.
        6. Execute `train_eval_deep_ens()` using either a saved model, the implicitly selected best model, or custom parameters.
        7. Directly access the saved models (via `self.best` or `get_saved_model()`) to generate plots or run predictions.
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
        Executes grid or random search training and evaluation across combinations of hyperparameters and/or features.

            - Clears the previous evaluation state.
            - Generates all unique combinations of the provided `params` dictionary.
            - Trains a base model for every combination using the underlying `Trainer`.
            - Passes all trained models to `ModelEval` to rank them by standard metrics (R2, RMSE, etc.).
            - Displays the "best" model for each metric category.
            - (Optional) Automatically saves the model that performed best in a specified metric.

        :param model_name:      String name from available models (e.g., "MLPRegression", "RidgeRegression").
        :param dataset:         Pandas DataFrame containing the data, or a string referencing a built-in dataset ("CHF").
        :param features_list:   List of a list of features from ModelData
                                EXAMPLE:
                                    [["temp", "humidity", "wind_speed"], ["temp", "humidity"]]
        :param params:          Parameters for the chosen model, with keys matching model's parameter kwargs.
                                Values must be passed as a list of parameters to try.
                                EXAMPLE:
                                    `{"alpha": [0.1, 0.2, 0.3], "hidden_layer_sizes": [(100, 100), (200, 200)]}`
                                NOTES:
                                    - If being used: train_test_split must be entered as {..., "tts": [0.2], ...}
                                    - Single value parameters must be wrapped in a list (e.g. {"tts": 0.2} becomes {"tts": [0.2]})
        :param target_cols:     List of target column names (e.g., ["CHF"]).
        :param plot_func:       Optional callable plotting function to attach to the trained models.
        :param random_state:    Integer for reproducible train/test splits and weight initialization, or None for true randomness.
        :param random_search:   Integer limiting the maximum number of models to train (randomly sampled from the grid).
        :param display_best:    Boolean flag to print the evaluation summary and parameters of the top-performing models.
        :param autosave:        String name of the evaluation metric (e.g., "R2", "95 CI"). The model scoring best
                                in this metric will automatically be saved to memory as `self.best`.
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
                            random_state=None,
                            auto_save: str | bool = False,
                            ):
        """
        Trains and evaluates a Deep Ensemble model using uncertainty quantification.

        Workflow & Selection Methods:
            There are three distinct ways to configure the base architecture for the ensemble members.
            The method dynamically identifies which workflow to use based on the arguments provided:
            1. Saved Index (`best_idx` is passed): Pulls a fully configured model previously stored via `save_best()`.
            2. Implicit Best (`base_model` and `best_idx` are None): Automatically uses the `self.best` model currently active.
            3. Custom Configuration: Manually provide `base_model`, `features`, `target_cols`, and `params`.

        :param dataset:         Pandas DataFrame or string referencing a built-in dataset.
        :param best_idx:        Integer index of a previously saved model to use as the base architecture template.
        :param base_model:      String name of the architecture to use (e.g., "MLPRegression"). Required if building from scratch.
        :param target_cols:     List of target column names. Required if building from scratch.
        :param n_models:        Number of member models to train inside the ensemble.
        :param features:        List of lists of feature column names. Required if building from scratch.
        :param params:          Dictionary of hyperparameters. Required if building from scratch.
        :param plot_func:       Optional callable plotting function.
        :param random_state:    Integer for reproducibility, or None for true randomness. Member models receive offset seeds.
        :param auto_save:       String specifying which metric's best ensemble to save (e.g., "R2"), or a boolean to save the first result.
        """
        ### Figure out how model params were passed into function ###
        model: IModel | None = None
        params_ = params

        # Model input 1: Model by saved index
        if best_idx is not None:
            model = self.get_saved_model(best_idx)

        # Model input 2: Model inferred from autosaved or selected best model
        elif not base_model:
            if self.best:
                model = self.best
            else:
                raise ValueError("ModelMaker.train_deep_ensemble() requires a selected best model if no base_model is provided.")

        # Model input 3: Reused a previously trained model's parameters
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

        if n_models:
            params_["n_models"] = n_models

        ### Train Deep Ensemble ###
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

        ### Evaluate ###
        self.eval.add_models(deep_ens_s)

        # multiple models
        if len(deep_ens_s) > 1:
            self.best_models = self.eval.evaluate()
            self.eval.display_best()

        # single model
        else:
            self.best_models = deep_ens_s
            self.eval.display_best(deep_ens_s[0])

        # autosave
        if auto_save:
            if isinstance(auto_save, str):
                self.save_best(self.eval.best_models.get(auto_save))
            else:
                self.save_best(0)

    def save_best(self, idx_or_model: int | IModel) -> IModel:
        """
        Saves a model into persistent memory within the ModelMaker instance.

        This sets the target model to `self.best` for immediate access and appends it to
        `self._saved_models` for safe retrieval across multiple workflow stages.

        :param idx_or_model:    Either an integer index corresponding to a model in the currently evaluated
                                `best_models` list, or a direct `IModel` object.

        Example usage after choosing best model, directly access model's methods:
            ModelMaker.best.plot()
            scores: dict = ModelMaker.best.get_scores()
        """
        try:
            # grab model from best_models after running train_and_eval()
            if isinstance(idx_or_model, int):
                self.best = self.best_models[idx_or_model]
            # manually pass in model
            else:
                self.best = idx_or_model

            # save the model for later
            print(f"Saved {self.cur_model_name} at index {len(self._saved_models)}.")
            self._saved_models.append(self.best)

            return self.best
        except Exception as e:
            print(f"ModelMaker.save_best() failed! "
                  f"No Model was saved; current number of saved models: {len(self._saved_models)}.\n"
                  "\tIf using an index, ensure train_and_eval() has been called, and index is valid.\n"
                  "\tIf passing in an IModel, ensure the model is a valid IModel\n", e)

    def clear_evaluator(self):
        """
        Clears the internal ModelEval state, emptying the list of currently ranked models.
        This should be called between separate isolated training runs to prevent old models
        from polluting new evaluation results.
        """
        self.eval.clear_all()

    def get_saved_model(self, idx: int) -> IModel | None:
        """
        Retrieves a previously saved model from the internal history.

        :param idx: Integer index corresponding to the chronological order the model was saved.
        :return:    The requested IModel object, or None if the index is out of bounds.
        """
        try:
            return self._saved_models[idx]
        except IndexError as e:
            print("get_saved_model() index out of range!\n", e)
            return None
