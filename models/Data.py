import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

# sklearn
from sklearn.linear_model import *
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score



ROOT = Path(__file__).resolve().parent.parent
WEATHER_PATH = ROOT / "data" / "DRO_2025-01-01_2026-02-09.csv"
ENERGY_PATH = ROOT / "data" / "SunnysideTotalPower2025-01-01_2026-02-12.csv"

"""
——— Class ModelData ———

Contains 2 data attributes for training models. The dataframes are already cleaned and reorganized.
    - weather:  dataframe of weather data
    - energy:   dataframe of energy data
    
NOTE: feel free to add some methods for changing the dataframes if needed!

"""
class ModelData:

    def __init__(self):
        # get data
        try: self.weather = pd.read_csv(WEATHER_PATH)
        except FileNotFoundError: print("No weather data found @ ", WEATHER_PATH)
        try: self.energy = pd.read_csv(ENERGY_PATH)
        except FileNotFoundError: print("No energy data found @ ", ENERGY_PATH)

        # set datetime and indices
        self.energy["timestamp"] = pd.to_datetime(self.energy["timestamp"])
        self.weather["datetime"] = pd.to_datetime(self.weather["datetime"])
        self.weather.set_index("datetime", inplace=True)
        self.energy.set_index("timestamp", inplace=True)

        # reorganize energy data
        self.energy = -self.energy.resample('h').mean()
        # print(self.energy.loc["2025-01-01":"2026-01-01"].sum())

        # drop last days to match and remove forcast hours
        self.energy, self.weather = self.energy.align(self.weather, join="inner", axis=0)

        # print(self.weather.shape, self.energy.shape)

    def get_or_do_something__(self):
        # TODO
        pass


if __name__ == "__main__":
    data = ModelData()