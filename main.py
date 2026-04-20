from Models.ModelMaker import ModelMaker
from util.util import feature_combo
from util.plots import *
from util.arg_parser import parse_args_save_paths


if __name__ == "__main__":

    parse_args_save_paths()

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


