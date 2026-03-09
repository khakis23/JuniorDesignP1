import tensorflow as tf
from Models.src.IModel import IModel
from Models.src.Trainer import Trainer
from Models.src.ModelEval import ModelEval


"""
Wrapper class. Create a single model maker per number of final models wanted. 
This is the only class and file that ie required to train and compare all models.
"""
class ModelMaker:

    def __init__(self):
        self.trainer = Trainer()
        self.eval = ModelEval()
        self.best: IModel = None
        self.best_models = []

        devices = tf.config.list_physical_devices()
        print(f"Available Processors: {devices}")  # if GPU appears, then it will be used

    def train_and_eval(self, model_name: str, features_list: list[list[str]], params: dict[str, list], random_state=None):
        """
        Train and evaluate a combination of a chosen model.

        :param model_name:      String name from available models (see Trainer for list of models)
        :param features_list:   List of list of features from ModelData

                                ()

        :param params:          Parameters for the chosen model, with keys matching model's parameter kwargs, and
                                values as a list of parameters to try.
                                EXAMPLE:
                                    `{"alpha": [0.1, 0.2, 0.3], "hidden_layer_sizes": [(100, 100), (200, 200)]}`
                                NOTES:
                                    - If being used: train_test_split must be entered as {..., "tts": [0.2], ...}
                                    - Single value parameters must be wrapped in a list (e.g.  {"tts": 0.2} becomes {"tts": [0.2]})
        :param random_state:    Random state for train_test_split and confidence interval
        """
        # train and evaluate
        self.eval.add_models(self.trainer.test_train(model_name, features_list, params, random_state=random_state))
        self.best_models = self.eval.evaluate()


        # display best models
        self.eval.display_best()


    def choose_best(self, idx: int):
        """
        Set the best model to self.best, which give public access to the best model object.

        :param idx: index of the models that have displayed using train_and_eval()

        Example usage after choosing best model::

            ModelMaker.best.plot()
            scores: dict = ModelMaker.best.get_scores()

        """
        if not self.best_models:
            raise IndexError("ModelMaker.choose_best() cannot be called before train_and_eval().")
        self.best = self.best_models[idx]
