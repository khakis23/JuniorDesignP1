import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent.parent
WEATHER_PATH = ROOT / "data" / "DRO_2025-01-01_2026-02-09.csv"
ENERGY_PATH = ROOT / "data" / "SunnysideTotalPower2025-01-01_2026-02-12.csv"

"""
——— Class ModelData ———

Contains 2 data attributes for training models. The dataframes are already cleaned and reorganized.
    - weather:  dataframe of weather data
    - energy:   dataframe of energy data
    
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

        # drop last days to match and remove forcast hours
        self.energy, self.weather = self.energy.align(self.weather, join="inner", axis=0)

        # add custom features to self.features
        self.features = self.weather.copy()
        self._engineer_features()

    def _engineer_features(self):
        ### Cyclic Time Features ###
        hour = self.features.index.hour
        day = self.features.index.dayofyear

        # hour of day sin/cos
        self.features["hsin"] = np.sin(2 * np.pi * hour / 24)
        self.features["hcos"] = np.cos(2 * np.pi * hour / 24)

        # day of year sin/cos
        self.features["dsin"] = np.sin(2 * np.pi * day / 365.25)  # (leap year)
        self.features["dcos"] = np.cos(2 * np.pi * day / 365.25)

        ### Derivative Features ###
        # create features
        self.features["dtemp"] = self.features["temp"].diff()
        self.features["dwindspeed"] = self.features["windspeed"].diff()
        self.features["dcloudcover"] = self.features["cloudcover"].diff()

        self.features["dsolarradiation"] = self.features["solarradiation"].diff()
        self.features["dsolarenergy"] = self.features["solarenergy"].diff()

        # zero NaNs in derivative features
        num_cols = self.features.select_dtypes(include=["number"]).columns
        self.features[num_cols] = self.features[num_cols].fillna(0)


if __name__ == "__main__":
    data = ModelData()
