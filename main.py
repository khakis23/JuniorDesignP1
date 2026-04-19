from Models.ModelMaker import ModelMaker
from util.util import feature_combo
from util.plots import *


if __name__ == "__main__":

    features = feature_combo(["D", "L", "P", "G",  "Tin", "Xe"], 4, 6)
    print("Total Features:", len(features))

    mm = ModelMaker()

    mm.train_and_eval(
        "MLPRegression",
        "CHF",
        features_list=[[ "D",  "P",  "G",  "Tin",  "Xe"]],
        params={
            "tts": 0.2,
            "activation": ["relu"],
            "hidden_layer_sizes": [(128, 64, 32)],
            "learning_rate": [0.001],
            "l2_alpha": [.0001],
            "epochs": 50,
            "early_stopping": True,
        },
        plot_func=plot_actual_vs_pred,
        autosave="R2",
        # random_search=15,
    )

    # pick = int(input("Enter and index: ").strip())
    # mm.save_best(pick)

