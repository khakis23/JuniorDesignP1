import random
from Models import ModelMaker
from util.util import feature_combo


if __name__ == "__main__":

    ### EXAMPLE CODE ###

    mm = ModelMaker()

    features = feature_combo(["temp", "cloudcover", "solarradiation", "dcloudcover", "dtemp", "dsolarradiation", "solarenergy", "windspeed", "hcos", "hsin", "celltemp"], 3, 11)
    print("Total Features:", len(features))

    random_features = [random.choice(features) for _ in range(10_000)]
    # print("Random Features:", random_features)

    """ RIDGE """
    #
    all_feat = feature_combo(["temp", "cloudcover", "solarradiation", "dcloudcover", "dtemp", "dsolarradiation", "solarenergy", "windspeed", "hcos", "hsin", "heatloss"], 3, 6)


    # mm.train_and_eval("RidgeRegression",
    #                   random_features,
    #                   {
    #                       "tts": [0.2, 0.25],
    #                       "alphas": [np.logspace(-3, 3, 100)],  # in a list cuz ridgeCV already cross validates by itself
    #                    }, random_state=42)
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
    #
    # mm.train_and_eval("LSTMRegression",
    #                   [
    # #                       # ["cloudcover",  "solarradiation",  "dcloudcover",  "dtemp",  "dsolarradiation",  "solarenergy",  "hcos",  "heatloss"],
    # #                       # ["cloudcover", "solarradiation", "dcloudcover", "dtemp", "dsolarradiation", "solarenergy", "hcos", "heatloss", "temp"],
    #                       ["cloudcover", "solarradiation", "dcloudcover", "dtemp", "dsolarradiation", "solarenergy", "hcos", "celltemp", "temp"],
    # #                       # ["cloudcover", "solarradiation", "dcloudcover", "dtemp", "dsolarradiation", "solarenergy", "hcos", "celltemp"],
    # #                       # ["temp",  "solarradiation",  "dcloudcover",  "dsolarradiation",  "hcos"]
    #                   ],
    #                   {
    #                         "tts": [0.2],
    #                         "lookback": [48],
    #                         "epochs": [500],             # High cap, relying on EarlyStopping to halt at the optimal point
    #                         "batch_size": [32],      # Smaller batches often help models escape local minima in tabular data
    #                         "lstm_units_1": [256, 128, 64],        # Reduced from 128 to prevent the model from memorizing noise
    #                         "lstm_units_2": [128, 64, 32],        # Reduced from 64 for a tighter information bottleneck
    #                         "lstm_units_3": [64, 32, 16],
    #                         "dense_units": [16],         # Simplified final layer
    #                         "dropout_rate": [0.3],  # Standard regularization to drop overly aggressive nodes
    #                         "validation_split": [0.1, 0.2]   # Consistent holdout size for validation loss tracking
    #                   }, random_state=42)


    """ MLP """
    # mm.train_and_eval("MLPRegression",
    #                   [
    #                       # ["cloudcover",  "solarradiation",  "dcloudcover",  "dtemp",  "dsolarradiation",  "solarenergy",  "hcos",  "heatloss"],
    #                       ["cloudcover", "solarradiation", "dcloudcover", "dtemp", "dsolarradiation", "solarenergy", "hcos", "heatloss", "temp"],
    #                       ["cloudcover", "solarradiation", "dcloudcover", "dtemp", "dsolarradiation", "solarenergy",
    #                        "hcos", "celltemp", "temp"],
    #                       ["cloudcover", "solarradiation", "dcloudcover", "dtemp", "dsolarradiation", "solarenergy", "hcos", "celltemp"],
    #                       # ["temp",  "solarradiation",  "dcloudcover",  "dsolarradiation",  "hcos"]
    #                   ],
    #                   {
    #                       "tts": [0.2],
    #                       "activation": ["relu" ],
    #                       "learning_rate_init": [0.001, 0.005],
    #                       "hidden_layer_sizes": [(64, 32), (128, 64, 32), (256, 128, 64, 32)],
    #                       "alpha": [0.001, 0.01, 0.1],
    #                       "max_iter": [800],
    #                       "early_stopping": [True],
    #                       "batch_size": [32, 64],
    #                   }, random_state=42
    #                   )

    """ Random Forest """
    # mm.train_and_eval("RandomForestRegression",
    #                     [
    #                         # ["cloudcover",  "solarradiation",  "dcloudcover",  "dtemp",  "dsolarradiation",  "solarenergy",  "hcos",  "heatloss"],
    #                         # ["cloudcover", "solarradiation", "dcloudcover", "dtemp", "dsolarradiation", "solarenergy", "hcos", "heatloss", "temp"],
    #                         ["cloudcover", "solarradiation", "dcloudcover", "dtemp", "dsolarradiation", "solarenergy",
    #                          "hcos", "celltemp", "temp"],
    #                         ["cloudcover", "solarradiation", "dcloudcover", "dtemp", "dsolarradiation", "solarenergy", "hcos", "celltemp"],
    #                         # ["temp",  "solarradiation",  "dcloudcover",  "dsolarradiation",  "hcos"]
    #                     ],
    #                   {
    #                       "tts": [0.2],
    #                       "n_estimators": [100, 200, 600],
    #                       "max_depth": [15, 20],
    #                       "min_samples_split": [2, 3],
    #                       "min_samples_leaf": [1, 2, 4],
    #                       "max_features": ["sqrt", 0.5],
    #                       "bootstrap": [True]
    #                   },
    #                   random_state=42
    #                   )
    features = [
        ["temp", "cloudcover", "dtemp", "dsolarradiation", "windspeed", "hcos", "hsin", "celltemp"],
        ["cloudcover", "solarradiation", "dtemp", "dsolarradiation", "windspeed", "hcos"],
        ["temp", "cloudcover", "solarradiation", "dtemp", "dsolarradiation", "solarenergy", "hcos", "hsin", "celltemp"],
        ["cloudcover", "dcloudcover", "hsin"],
        ["temp", "cloudcover", "solarradiation", "dcloudcover", "dtemp", "dsolarradiation", "solarenergy", "windspeed",
         "hcos", "hsin", "celltemp"],
        ["cloudcover", "solarradiation", "dcloudcover", "dtemp", "dsolarradiation", "solarenergy", "hcos", "celltemp"],
        ["cloudcover", "solarradiation", "dcloudcover", "dtemp", "dsolarradiation", "solarenergy", "hcos", "celltemp",
         "temp"],
        ["cloudcover", "solarradiation", "dcloudcover", "dtemp", "dsolarradiation", "solarenergy", "hcos", "heatloss",
         "temp"],
        ["cloudcover", "solarradiation", "dcloudcover", "dtemp", "dsolarradiation", "solarenergy", "hcos", "heatloss"],
        ["temp", "solarradiation", "dcloudcover", "dsolarradiation", "hcos"],
        ["cloudcover", "dtemp", "dsolarradiation", "windspeed", "hcos", "hsin"],
        ["temp", "solarradiation", "sunelevation", "cloudcover", "sunazimuth", "solarenergy"],
    ]

    """ Gradient Boosting """
    mm.train_and_eval("GradientBoostingRegression",
                        features[6:8], {
                          "tts": [0.2],
                          "n_estimators": [25, 50, 75, 100, 125],
                          "learning_rate": [0.05, 0.1],
                          "max_depth": [3, 5],
                          "min_samples_split": [100, 200],
                          "min_samples_leaf": [20, 40],
                          "subsample": [0.8],
                          "validation_fraction": [0.1],
                          "n_iter_no_change": [10, 20],
                          "tol": [1e-3, 1e-4],},
                      random_state=42)

