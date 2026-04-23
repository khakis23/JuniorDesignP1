from Models.ModelMaker import ModelMaker
from util.util import feature_combo
from util.plots import *
from util.arg_parser import parse_args_save_paths


if __name__ == "__main__":
    parse_args_save_paths()

    features = feature_combo(['PSZ', 'DOM', 'vanA', 'vanB', 'subcool', 'CRD', 'flow_rate', 'power_density', 'VFNGAP'], 5, 9)
    params = {
        "hidden_layer_sizes": [
            (128, 64, 32),
            (256, 128, 64, 32),
            (512, 256, 128, 64),
            (512, 512, 256, 128),
            (512, 512, 256, 128, 64),
        ],
        "learning_rate": [0.0002, 0.0005, 0.0008],
        "l2_alpha": [0.001, 0.01, 0.05, 0.1],
        "activation": ["swish", "gelu"],
        "batch_size": [32, 64],
        "log_var_min": [-7.0, -5.0],
        "log_var_max": [5.0, 3.0],
        "early_stopping": [1],
        "epochs": [1000],
        "tts": [0.2],
        "folds": [4],
    }

    mm = ModelMaker()
    mm.train_and_eval(
        "MLPRegression",
        "BWR",
        features_list=features,
        params=params,
        autosave="CV R2",
        plot_func=plot_actual_vs_pred,
        random_state=42,
        random_search=250,
    )
    mm.train_eval_deep_ens(
        "BWR",
        plot_func=deep_ensemble_wrapper,
        n_models=6,
    )
    mm.train_eval_deep_ens(
        "BWR",
        plot_func=deep_ensemble_wrapper,
        n_models=50,
    )
