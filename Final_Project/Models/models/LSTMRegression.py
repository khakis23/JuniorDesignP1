import numpy as np
import pandas as pd

from keras import Input
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split, TimeSeriesSplit, cross_val_score

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.layers import Bidirectional, Dropout, Input, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

from Data.ModelData.ModelData import ModelData
from Models.src.IModel import IModel
from util.util import plot_daily_solar_output


"""
Implementation of IModel using TensorFlow's LSTMReg model.

~60% of this file is writen my an LLM.
"""
class LSTMRegression(IModel):

    def __init__(self, features: list[str], data: ModelData):
        super().__init__(features, data)
        # hours of data that can be looked back into
        self.lookback: int

        # scale
        self.x_scaler = StandardScaler()
        self.y_scaler = StandardScaler()

    def _create_sequences(self, x_data: np.ndarray, y_data: np.ndarray):
        x_seq, y_seq = [], []
        for i in range(len(x_data) - self.lookback):
            # Add +1 to the slice so the current timestep's features are included in the sequence
            x_seq.append(x_data[i: i + self.lookback + 1])
            y_seq.append(y_data[i + self.lookback])
        return np.array(x_seq), np.array(y_seq)

    def train_and_fit(self, tts: float = 0.0, random_state=42, **kwargs):
        # having issues with random states (likely GPU related?)
        self._random_state = random_state
        np.random.seed(random_state)
        tf.random.set_seed(random_state)

        # This will slow down training
        # tf.config.experimental.enable_op_determinism()

        # Custom Train/Test Split
        if tts > 0:
            self._test_size = tts
            self._random_state = random_state

            self._x["train"], self._x["test"], self._y["train"], self._y["test"] = train_test_split(
                self._x["full"], self._y["full"], test_size=tts, shuffle=False  # False for lookback
            )

        # Extract Keras-specific training parameters from kwargs
        epochs = kwargs.get("epochs", 300)
        batch_size = kwargs.get("batch_size", 64)
        lstm_units_1 = kwargs.get("lstm_units_1", 128)
        lstm_units_2 = kwargs.get("lstm_units_2", 64)
        lstm_units_3 = kwargs.get("lstm_units_3", 32)
        dense_units = kwargs.get("dense_units", 32)
        dropout_rate = kwargs.get("dropout_rate", 0.3)
        val_split = kwargs.get("validation_split", 0.15)
        self.lookback = kwargs.get("lookback", 24)

        # Fit Scalers and Transform Training Data
        x_train_scaled = self.x_scaler.fit_transform(self._x["train"])
        y_train_2d = self._y["train"].values.reshape(-1, 1)
        y_train_scaled = self.y_scaler.fit_transform(y_train_2d).flatten()

        # create 3D sequences from data: VERY IMPORTANT!
        x_train_seq, y_train_seq = self._create_sequences(x_train_scaled, y_train_scaled)

        # Build Keras LSTM Architecture using the kwargs
        tf.random.set_seed(random_state)
        self.model = Sequential([
            Input(shape=(x_train_seq.shape[1], x_train_seq.shape[2])),

            Bidirectional(LSTM(lstm_units_1, return_sequences=True, activation='tanh')),
            BatchNormalization(),
            Dropout(dropout_rate),

            LSTM(lstm_units_2, return_sequences=True, activation='tanh'),
            Dropout(dropout_rate),

            LSTM(lstm_units_3, activation='tanh'),
            Dropout(dropout_rate),

            Dense(dense_units, activation='relu'),
            Dense(1)
        ])

        callbacks = [
            EarlyStopping(monitor='val_loss', patience=12, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-6)
        ]

        self.model.compile(optimizer='adam', loss=tf.keras.losses.Huber())

        # 6. Fit using the kwargs
        self.model.fit(
            x_train_seq, y_train_seq,
            validation_split=val_split,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=0
        )

        # Keep track of parameters for print_parameters()
        self._parameters = {
            "Lookback": self.lookback,
            "Epochs": epochs,
            "Batch Size": batch_size,
            "LSTM 1": lstm_units_1,
            "LSTM 2": lstm_units_2,
            "LSTM 3": lstm_units_3,
            "Dropout": dropout_rate,
            "Dense Units": dense_units,
            "Validation Split": val_split,
            "Test Size": self._test_size
        }

    def predict(self, x: pd.DataFrame = None) -> np.ndarray:
        if self.model is None:
            raise ValueError("Model has not been trained!")

        # Determine which dataset we are predicting on
        target_x = self._x["test"] if self._test_size else (self._x["full"] if x is None else x)
        target_y = self._y["test"] if self._test_size else self._y["full"]

        # scale and sequence
        x_scaled = self.x_scaler.transform(target_x)
        x_seq, _ = self._create_sequences(x_scaled, np.zeros(len(x_scaled)))

        # predict (outputs are scaled)
        preds_scaled = self.model.predict(x_seq, verbose=0)
        preds = self.y_scaler.inverse_transform(preds_scaled).flatten()

        ### Pad the predictions array ###
        # Because the LSTM looks back 'N' hours, it cannot predict the first 'N' hours of the test set.
        # Pad the beginning with NaNs so the predictions array perfectly matches the original index length.
        padded_preds = np.full(len(target_x), np.nan)
        padded_preds[self.lookback:] = preds

        self._predictions = padded_preds
        self._clamp_predictions()

        if self._test_size:
            self._score()

        return self._predictions

    def _score(self, folds: int = 0):
        # Slice out the NaN padding from the lookback window
        valid_idx = ~np.isnan(self._predictions.ravel())

        y_true_valid = self._y["test"][valid_idx]
        y_pred_valid = self._predictions.ravel()[valid_idx]
        y_pred_c_valid = self._c_predictions.ravel()[valid_idx]

        # Standard holdback (no CV)
        self._scores = {
            "R2": r2_score(y_true_valid, y_pred_valid),
            "RMSE": mean_squared_error(y_true_valid, y_pred_valid) ** 0.5,
            "RMSE Clamped": mean_squared_error(y_true_valid, y_pred_c_valid) ** 0.5,
            "MAE": mean_absolute_error(y_true_valid, y_pred_valid),
            "CI": self._get_bootstrap_safe(y_true_valid, y_pred_valid),
            # "Epochs": self.model.hi  # TODO
        }


    def _get_bootstrap_safe(self, y_true, y_pred):
        """Helper to run bootstrap without NaN values"""
        original_y = self._y["test"]
        original_preds = self._predictions

        self._y["test"] = y_true
        self._predictions = y_pred

        ci = self._get_bootstrap(r2_score)

        self._y["test"] = original_y
        self._predictions = original_preds
        return ci


    def plot(self):
        if self._test_size:
            #  Safely grab the CI and ensure it's a tuple/list before indexing
            ci = self._scores.get("CI", (0.0, 0.0))
            ci_display = f"[{round(ci[0], 2)} : {round(ci[1], 2)}]" if isinstance(ci, (list, tuple)) else str(round(ci, 2))

            display_features = {
                "R2": round(self._scores["R2"], 3),
                "CI": ci_display,
                "RMSE": round(self._scores["RMSE Clamped"], 2),
                "Lookback": self.lookback,
            }

            plot_daily_solar_output(
                self._c_predictions,
                self._y["test"],
                "LSTM Regression",
                display_features,
            )