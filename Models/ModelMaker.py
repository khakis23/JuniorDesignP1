import tensorflow as tf
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
        self.eval: ModelEval
        self.best: IModel = None
        self.best_models = []
        self.cur_model_name: str = "NULL"
        self._saved_models = []

        devices = tf.config.list_physical_devices()
        print(f"Available Processors: {devices}")  # if GPU appears, then it will be used

    def train_and_eval(self,
                       model_name: str,
                       features_list: list[list[str]],
                       params: dict[str, list],
                       random_state=None,
                       random_search: float=0.0,
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
        self.eval = ModelEval()

        # train and evaluate
        self.eval.add_models(self.trainer.test_train(model_name, features_list, params, random_state=random_state))
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
            self.best.print_scores()

    def train_deep_ensemble(self, **kwargs):
        # get hyperparams from existing model
        if best_idx := kwargs.get("best_idx", None):
            model = self.get_saved_model(best_idx)
            params = model.get_parameters()
            features = model.get_features()

        # get hyperparams from kwargs
        else:
            params = kwargs.get("hp", None)
            features = kwargs.get("f", None)

        if not params: raise ValueError("`hp` kwarg is missing or model is missing hyperparameters.")
        if not features: raise ValueError("`f` kwarg is missing or model is missing features.")

        # Trainer will handle trinaing deep ensemble
        deep_ens = self.trainer.test_train("DeepEnsemble", features, params)
        # TODO do something, maybe just use reuse train_and_eval() depending on how we are going to evaluate the model

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

    def get_saved_model(self, idx: int) -> IModel | None:
        try:
            return self._saved_models[idx]
        except IndexError as e:
            print("get_saved_model() index out of range!\n", e)
            return None
