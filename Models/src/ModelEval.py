import numpy as np

from Models.src.IModel import IModel
from util.SavePaths import SavePaths
from abc import ABC


"""
Implement in conjunction with IModel to evaluate many models. display_results() will cleanly
display the best models.
"""


def _display(model: IModel, count: int=-1, score_name: str=""):
    if score_name:
        print(f"\n\n======== Model {count} | Best {score_name} ========")
    model.plot()
    print("\n————— Scoring —————")
    if SavePaths.save_results:
        model.write_scores_to_file(SavePaths.save_path / f"{model.__class__.__name__}_results.txt")
    model.print_scores()
    print("\n ————— Model Parameters —————")
    if SavePaths.save_results:
        model.write_parameters_to_file(SavePaths.save_path / f"{model.__class__.__name__}_results.txt")
    model.print_parameters()
    print("\n————— Model Features ————— \n", *model.features, sep="  ")


class ModelEval:

    def __init__(self):
        self.models: list[IModel] = []
        self.best_models = {}  # by score key  (e.g.  {"R2": <ModelReference>, ...}

    def add_models(self, model: list[IModel]):
        self.models.extend(model)

    def evaluate(self) -> list[IModel]:
        """
        Evaluate all models based on each score parameter

        :return: {"<score_name>": <model_reference>, ...>}  (e.g.  {"R2": <ModelReference>, ...}
        """
        def _check_ci(ci):
            # NOTE some modes return an array-like structure and some return a float.
            if isinstance(ci, tuple) or isinstance(ci, list) or isinstance(ci, np.ndarray):
                return np.mean(ci)
            return ci

        greater_than = ["R2", "CV R2", "95 CI"]
        less_than = ["RMSE", "MAE", "RMSE Clamped", "CI", "epSTD", "totalSTD"]

        print(f"Evaluating {len(self.models)} models...")

        # compare important features like R2, RMSE, etc., then add to self.best_models
        best_scores = self.models[0].get_scores()
        self.best_models = {sn: self.models[0] for sn in best_scores.keys()}

        # CUSTOM parameters adjustment
        if "CI" in best_scores:
            best_scores["CI"] = _check_ci(best_scores["CI"])

        for i, model in enumerate(self.models):
            if i == 0: continue  # already loaded first model into best_scores/best_models
            for sn, sv in model.get_scores().items():

                # CUSTOM parameters
                if sn == "CI":
                    sv = _check_ci(sv)

                # scores that are better when higher
                if sn in greater_than and sn in best_scores:
                    # if new best score
                    if sv > best_scores[sn]:
                        best_scores[sn] = sv
                        self.best_models[sn] = model

                # scores that better when lower
                if sn in less_than and sn in best_scores:
                    # print(f"{sv}: {sn}")  # TODO DEBUGGING
                    if sv < best_scores[sn]:
                        best_scores[sn] = sv
                        self.best_models[sn] = model

        return [x for x in self.best_models.values()]

    def display_best(self, model=None):
        """
        """
        # passed in single model to display
        if model:
            _display(model)
            return

        # display saved best evaluated model
        count = 0
        for sn, m in self.best_models.items():
            count += 1
            _display(m, count, sn)

    def clear_models(self):
        self.models = []

    def clear_all(self):
        self.clear_models()
        self.best_models = {}
