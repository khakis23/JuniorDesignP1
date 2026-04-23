from Data.ModelData.ModelData import ModelData
from Models.models.GradientBoostingRegression import GradientBoostingRegression
from Models.models.MLPRegression import MLPRegression
from Models.models.RandomForestRegression import RandomForestRegression
from Models.models.RidgeRegression import RidgeRegression
from Models.models.LSTMRegression import LSTMRegression
from Models.models.DeepEnsemble import DeepEnsemble
from datasets import data_handler as h


# Implemented models go here
MODEL_FACTORY = {
    "RidgeRegression": RidgeRegression,  # TODO NOT UPDATED
    "LSTMRegression": LSTMRegression,    # TODO NOT UPDATED
    "MLPRegression": MLPRegression,
    "RandomForestRegression": RandomForestRegression,          # TODO NOT UPDATED
    "GradientBoostingRegression": GradientBoostingRegression,  # TODO NOT UPDATED
    "DeepEnsemble": DeepEnsemble,
}
# Built-in datasets go here:
#   - must return (features, targets, data, <optional>data_test)
DATABASE_FACTORY = {
    "CHF": h.load_chf,
    "Reactor": h.load_reactor,
    "BWR": h.load_bwr,
}