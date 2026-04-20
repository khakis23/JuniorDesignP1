from Models.ModelMaker import ModelMaker
from util.util import feature_combo
from util.plots import *
from util.arg_parser import parse_args_save_paths


if __name__ == "__main__":
    parse_args_save_paths()

    features = feature_combo(["D", "L", "P", "G",  "Tin", "Xe"], 4, 6)
    params = {
        "tts": [0.2],
        "log_var_min": [-10.0],
        "log_var_max": [10.0],
        "validation_split": [0.1],
        "early_stopping": [1],
        "epochs": [1000],
        "hidden_layer_sizes": [
            (64, 32),
            (128, 64, 32),
            (256, 128, 64, 32),
            (128, 128, 128)
        ],
        "learning_rate": [0.01, 0.001, 0.0001],
        "batch_size": [32, 64, 128],
        "l2_alpha": [0.0, 0.0001, 0.001],
        "activation": ["relu", "elu", "swish"]
    }

    mm = ModelMaker()
    mm.train_and_eval(
        "MLPRegression",
        "CHF",
        features_list=features,
        params=params,
        autosave="R2",
        plot_func=plot_actual_vs_pred,
        random_state=42,
        random_search=200
    )
    mm.train_eval_deep_ens(
        "CHF",
        plot_func=deep_ensemble_wrapper,
        n_models=6,
    )
    mm.train_eval_deep_ens(
        "CHF",
        plot_func=deep_ensemble_wrapper,
        n_models=100,
    )
