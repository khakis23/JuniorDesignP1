import numpy as np
from Models.ModelMaker import ModelMaker
from util.util import feature_combo


if __name__ == "__main__":

    ### EXAMPLE CODE ###

    mm = ModelMaker()

    """ RIDGE """
    #
    # all_feat = feature_combo(["temp", "cloudcover", "solarradiation", "dcloudcover", "dtemp", "dsolarradiation", "solarenergy", "windspeed", "hcos", "hsin", "heatloss"], 3, 6)
    #
    #
    # mm.train_and_eval("RidgeRegression",
    #                   all_feat,
    #                   {
    #                       "tts": [0.2, 0.25],
    #                       "alphas": [np.logspace(-3, 3, 50)],  # in a list cuz ridgeCV already cross validates by itself
    #                    })
    #
    # mm.choose_best(0)
    # print(mm.best.get_parameters())
    #
    # mm.best.train_and_fit()
    # mm.best.predict()   # TODO forcast data
    #
    """ LSTM 
    (this model requires tensor flow, and recommended to run on GPU — the console will log which processor(s) is available) 
    """
    # mm.train_and_eval("LSTMRegression",
    #                   [
    #                       ["temp", "cloudcover", "solarradiation", "dcloudcover", "dtemp", "dsolarradiation", "solarenergy", "windspeed", "hcos", "hsin", "heatloss"],
    #                       # ["heatloss", "cloudcover", "dtemp", "solarenergy", "windspeed", "hsin", "hcos"]
    #                       # ["cloudcover", "solarradiation", "hcos", "dtemp", "dsolarradiation"],
    #                       # ["temp", "cloudcover", "solarenergy", "solarradiation", "hsin", "hcos", "dtemp", "dwindspeed",
    #                       #  "dcloudcover", "dsolarradiation", "dsolarenergy"],
    #                       # ["temp", "cloudcover", "solarenergy", "hsin", "hcos", "dtemp", "dcloudcover",
    #                       #  "dsolarradiation"],
    #                       # ["temp", "cloudcover", "solarradiation", "hsin", "dtemp", "dcloudcover", "dsolarenergy"],
    #                       # ["temp", "cloudcover", "solarradiation", "hcos", "dtemp", "dcloudcover", "dsolarenergy"],
    #                       # ["temp", "solarradiation", "hcos", "dtemp", "dcloudcover", "dsolarradiation"],
    #                   ],
    #                   {
    #                         "tts": [0.2],
    #                         "lookback": [72],        # 2 to 3 full days gives the LSTM a proper cyclical baseline
    #                         "epochs": [300],             # High cap, relying on EarlyStopping to halt at the optimal point
    #                         "batch_size": [32],      # Smaller batches often help models escape local minima in tabular data
    #                         "lstm_units_1": [64],        # Reduced from 128 to prevent the model from memorizing noise
    #                         "lstm_units_2": [32],        # Reduced from 64 for a tighter information bottleneck
    #                         "dense_units": [16],         # Simplified final layer
    #                         "dropout_rate": [0.5],  # Standard regularization to drop overly aggressive nodes
    #                         "validation_split": [0.15]   # Consistent holdout size for validation loss tracking
    #                   }, random_state=42)


    """ MLP """
    # mm.train_and_eval("MLPRegression",
    #                   [
    #                       ["temp", "cloudcover", "solarradiation", "dcloudcover", "dtemp", "dsolarradiation", "solarenergy", "windspeed", "hcos", "hsin", "heatloss"],
    #                       ["heatloss", "cloudcover", "dtemp", "solarenergy", "windspeed", "hsin", "hcos"],
    #                       # ["cloudcover", "solarradiation", "hcos", "dtemp", "dsolarradiation"],
    #                       # # ["temp", "cloudcover", "solarenergy", "solarradiation", "hsin", "hcos", "dtemp", "dwindspeed",
    #                       # #  "dcloudcover", "dsolarradiation", "dsolarenergy"],
    #                       # ["temp", "cloudcover", "solarenergy", "hsin", "hcos", "dtemp", "dcloudcover",
    #                       #  "dsolarradiation"],
    #                       # ["temp", "cloudcover", "solarradiation", "hsin", "dtemp", "dcloudcover", "dsolarenergy"],
    #                       # # ["temp", "cloudcover", "solarradiation", "hcos", "dtemp", "dcloudcover", "dsolarenergy"],
    #                       # ["temp", "solarradiation", "hcos", "dtemp", "dcloudcover", "dsolarradiation"],
    #                   ],
    #                   {
    #                       "tts": [0.25],
    #                       "activation": ["relu"],
    #                       "hidden_layer_sizes": [(64, 32), (128, 64, 32), (256, 128, 64, 32)],
    #                       "alpha": [0.0001, 0.001, 0.01, 0.1, 1, 10],
    #                       "max_iter": [1000],
    #                   }, random_state=42
    #                   )

    """ Random Forest """
    # mm.train_and_eval("RandomForestRegression",
    #                     [
    #                         ["temp", "cloudcover", "solarradiation", "dcloudcover", "dtemp", "dsolarradiation",
    #                          "solarenergy", "windspeed", "hcos", "hsin", "heatloss"],
    #                         ["heatloss", "cloudcover", "dtemp", "solarenergy", "windspeed", "hsin", "hcos"],
    #                     ],
    #                   {
    #                       "tts": [0.25],
    #                       "n_estimators": [100, 250, 500],
    #                       "max_depth": [None, 10, 20, 30],
    #                       "min_samples_split": [2, 5, 10],
    #                       "min_samples_leaf": [1, 2, 4],
    #                       "max_features": [1.0, "sqrt"],
    #                       "bootstrap": [True]
    #                   },
    #                   random_state=42
    #                   )
    #

    """ Gradient Boosting """
    # mm.train_and_eval("GradientBoostingRegressor",
    #                     [
    #           ["temp", "cloudcover", "solarradiation", "dcloudcover", "dtemp", "dsolarradiation",
    #                          "solarenergy", "windspeed", "hcos", "hsin", "heatloss"],
    #                         ["heatloss", "cloudcover", "dtemp", "solarenergy", "windspeed", "hsin", "hcos"],
    #                     ],
    #                   {"n_estimators": [100, 250, 500],
    #                    "learning_rate": [0.01, 0.05, 0.1],
    #                    "max_depth": [3, 4, 5],
    #                    "min_samples_split": [2, 5],
    #                    "min_samples_leaf": [1, 2, 4],
    #                    "subsample": [0.8, 1.0]
    #                    }, random_state=42
    #                   )

