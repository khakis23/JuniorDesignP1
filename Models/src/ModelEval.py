import numpy as np

from Models.src.IModel import IModel
from abc import ABC


"""
Implement in conjunction with IModel to evaluate many models. display_results() will cleanly
display the best models.
"""
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
        greater_than = ["R2", "CV R2"]
        less_than = ["RMSE", "MAE", "RMSE Clamped", "CI"]

        # compare important features like R2, RMSE, etc., then add to self.best_models
        best_scores = self.models[0].get_scores()
        self.best_models = {sn: self.models[0] for sn in best_scores.keys()}

        # CUSTOM parameters adjustment
        if isinstance(best_scores["CI"], tuple):
            best_scores["CI"] = abs(best_scores["CI"][1] - best_scores["CI"][0])

        for i, model in enumerate(self.models):
            if i == 0: continue  # already loaded first model into best_scores/best_models
            for sn, sv in model.get_scores().items():

                # CUSTOM parameters
                if sn == "CI" and isinstance(sv, tuple):
                    # NOTE occasionally LSTMReg returns the range (not sure how this happens at the moment), so this
                    #      is a patch to ensure nothing crashes.
                    sv = abs(sv[1] - sv[0])

                # scores that are better when higher
                if sn in greater_than:
                    # if new best score
                    if sv > best_scores[sn]:
                        best_scores[sn] = sv
                        self.best_models[sn] = model

                # scores that better when lower
                if sn in less_than:
                    if sv < best_scores[sn]:
                        best_scores[sn] = sv
                        self.best_models[sn] = model

        return [x for x in self.best_models.values()]

    def display_best(self):
        """
        """
        count = 0
        for sn, model in self.best_models.items():
            count += 1
            print(f"\n\n======== Model {count} | Best {sn} ========")
            model.plot()
            print("\n————— Scoring —————")
            model.print_scores()
            print("\n ————— Model Parameters —————")
            model.print_parameters()
            print("\n————— Model Features ————— \n", *model.features, sep="  ")
