from typing import Callable

import numpy as np
import tensorflow as tf
import pandas as pd
import platform

from Models.src.IModel import IModel


class MLPRegression(IModel):
    """
    TensorFlow/Keras mean-variance MLP regressor.

    This version predicts:
        - mu      : mean prediction
        - log_var : log variance prediction

    That allows a Deep Ensemble to capture:
        - aleatoric uncertainty  (average predicted variance)
        - epistemic uncertainty  (variance of member means)

    Important behavior:
        - predict(x) returns mean only, so the rest of the project can still work.
        - predict_mean_variance(x) returns (mu, var) for deep ensemble use.
    """

    def __init__(self, features: list[str], targets: list[str], data: pd.DataFrame, data_test: pd.DataFrame = None, plot_func: Callable=None):
        super().__init__(features, targets, data, data_test, plot_func)
        self.model: tf.keras.Model | None = None
        self.history = None
        self.scaler_mean_ = None
        self.scaler_std_ = None
        self.y_mean_ = None
        self.y_std_ = None
        self._tf_params = {}

    def train_and_fit(self, tts: float = 0.0, random_state=42, **kwargs):
        # Call super first to set up the splits
        super().train_and_fit(tts, random_state)

        # Extract kwargs
        hidden_layer_sizes = kwargs.pop("hidden_layer_sizes", (100, 100))
        if isinstance(hidden_layer_sizes, int):
            hidden_layer_sizes = (hidden_layer_sizes,)

        activation = kwargs.pop("activation", "relu")
        optimizer_name = kwargs.pop("optimizer", "adam")
        learning_rate = kwargs.pop("learning_rate", kwargs.pop("learning_rate_init", 0.001))
        epochs = kwargs.pop("epochs", kwargs.pop("max_iter", 200))
        batch_size = kwargs.pop("batch_size", 32)
        validation_split = kwargs.pop("validation_split", kwargs.pop("validation_fraction", 0.1))
        early_stopping = kwargs.pop("early_stopping", False)
        patience = kwargs.pop("patience", 15)
        min_delta = kwargs.pop("min_delta", 0.0)
        verbose = kwargs.pop("verbose", 0)
        l2_alpha = kwargs.pop("l2_alpha", kwargs.pop("alpha", 0.0))
        log_var_min = kwargs.pop("log_var_min", -10.0)
        log_var_max = kwargs.pop("log_var_max", 10.0)

        # remove hidden kwargs that are not used by the model
        for key in list(kwargs.keys()):
            if key.startswith("_"):
                kwargs.pop(key)

        if kwargs:
            unknown = ", ".join(sorted(kwargs.keys()))
            print("Unexpected MLPRegression kwargs: " + unknown)

        # Store parameters for retrieval later
        self._tf_params = {
            "hidden_layer_sizes": tuple(hidden_layer_sizes),
            "activation": activation,
            "optimizer": optimizer_name,
            "learning_rate": learning_rate,
            "epochs": epochs,
            "batch_size": batch_size,
            "validation_split": validation_split,
            "early_stopping": early_stopping,
            "patience": patience,
            "min_delta": min_delta,
            "l2_alpha": l2_alpha,
            "random_state": random_state,
            "log_var_min": log_var_min,
            "log_var_max": log_var_max,
        }

        # Determine which data to fit on based on tts and pre-split indicators
        x_key = "train" if (self._test_size > 0 or self._pre_split) else "full"
        y_key = x_key

        X_fit = np.asarray(self._x[x_key], dtype=np.float32)
        y_fit = np.asarray(self._y[y_key], dtype=np.float32).reshape(-1, len(self.targets))

        # Scale features
        self._fit_scaler(X_fit)
        X_fit_scaled = self._transform_features(X_fit)

        # Scale Targets (y)
        self.y_mean_ = y_fit.mean(axis=0)
        self.y_std_ = y_fit.std(axis=0)
        self.y_std_[self.y_std_ == 0] = 1.0

        y_fit_scaled = (y_fit - self.y_mean_) / self.y_std_

        # Build Model
        self.model = self._build_model(
            input_dim=X_fit_scaled.shape[1] if len(X_fit_scaled.shape) > 1 else 1,
            hidden_layer_sizes=hidden_layer_sizes,
            activation=activation,
            optimizer_name=optimizer_name,
            learning_rate=learning_rate,
            l2_alpha=l2_alpha,
        )

        # Callbacks
        callbacks = []
        if early_stopping:
            callbacks.append(
                tf.keras.callbacks.EarlyStopping(
                    monitor="val_loss" if validation_split > 0 else "loss",
                    patience=patience,
                    min_delta=min_delta,
                    restore_best_weights=True,
                )
            )

        fit_kwargs = {
            "x": X_fit_scaled,
            "y": y_fit_scaled,
            "epochs": epochs,
            "batch_size": batch_size,
            "verbose": verbose,
            "callbacks": callbacks,
        }

        if validation_split > 0 and len(X_fit_scaled) > 1:
            fit_kwargs["validation_split"] = validation_split

        # Train Model
        self.history = self.model.fit(**fit_kwargs)

        # Save parameters to base class property
        history_loss = self.history.history.get("loss", []) if self.history else []
        history_val_loss = self.history.history.get("val_loss", []) if self.history else []

        self._parameters = {
            # Arguments expected by train_and_fit
            "tts": self._test_size,
            "epochs": self._tf_params.get("epochs"),
            "hidden_layer_sizes": self._tf_params.get("hidden_layer_sizes"),
            "activation": self._tf_params.get("activation"),
            "optimizer": self._tf_params.get("optimizer"),
            "learning_rate": self._tf_params.get("learning_rate"),
            "batch_size": self._tf_params.get("batch_size"),
            "validation_split": self._tf_params.get("validation_split"),
            "early_stopping": self._tf_params.get("early_stopping"),
            "l2_alpha": self._tf_params.get("l2_alpha"),
            # "output_type": self._tf_params.get("output_type"),
            "log_var_min": self._tf_params.get("log_var_min"),
            "log_var_max": self._tf_params.get("log_var_max"),

            # Results (prefixed with underscores so we can ignore them in the loop)
            "_epochs_trained": len(history_loss),
            "_final_loss": float(history_loss[-1]) if history_loss else None,
            "_final_val_loss": float(history_val_loss[-1]) if history_val_loss else None,
        }

        # Do NOT call self._fit() as Keras models are already fit above.
        # If we need predictions initialized immediately:
        if self._test_size > 0 or self._pre_split:
            self.predict()  # This will populate self._predictions and call self._score()

    def _build_model(
            self,
            input_dim: int,
            hidden_layer_sizes: tuple[int, ...],
            activation: str,
            optimizer_name: str,
            learning_rate: float,
            l2_alpha: float,
    ) -> tf.keras.Model:
        regularizer = tf.keras.regularizers.l2(l2_alpha) if l2_alpha else None

        # Output layer dimensions dynamically derived from targets list length
        num_targets = len(self.targets)

        model = tf.keras.Sequential(name="mlp_mean_variance_regression")
        model.add(tf.keras.layers.Input(shape=(input_dim,)))

        for units in hidden_layer_sizes:
            model.add(
                tf.keras.layers.Dense(
                    units,
                    activation=activation,
                    kernel_regularizer=regularizer,
                )
            )

        # Output mu and log_var for EACH target
        model.add(tf.keras.layers.Dense(num_targets * 2, activation="linear"))

        optimizer = self._make_optimizer(optimizer_name, learning_rate)

        # We must wrap the custom loss function to ensure it has access to the instance variables
        def loss_wrapper(y_true, y_pred):
            # Split the tensor in half based on the number of targets
            mu = y_pred[:, :num_targets]
            log_var = y_pred[:, num_targets:]

            log_var = tf.clip_by_value(
                log_var,
                self._tf_params.get("log_var_min", -10.0),
                self._tf_params.get("log_var_max", 10.0),
            )

            precision = tf.exp(-log_var)

            # Sum NLL across the targets axis, then take the mean across the batch
            return tf.reduce_mean(tf.reduce_sum(0.5 * (log_var + tf.square(y_true - mu) * precision), axis=1))

        model.compile(optimizer=optimizer, loss=loss_wrapper)
        return model

    @staticmethod
    def _make_optimizer(name: str, learning_rate: float):
        # TF slow on Apple Silicon, so use legacy Adam
        if name == "adam" and platform.processor() == "arm":
            return tf.keras.optimizers.legacy.Adam(learning_rate=learning_rate)

        name = name.lower()
        optimizers = {
            "adam": tf.keras.optimizers.Adam,
            "sgd": tf.keras.optimizers.SGD,
            "rmsprop": tf.keras.optimizers.RMSprop,
            "adagrad": tf.keras.optimizers.Adagrad,
            "adamax": tf.keras.optimizers.Adamax,
            "nadam": tf.keras.optimizers.Nadam,
        }
        if name not in optimizers:
            raise ValueError(f"Unsupported optimizer '{name}'")
        return optimizers[name](learning_rate=learning_rate)

    def _fit_scaler(self, x: np.ndarray):
        # calculate mean and std
        self.scaler_mean_ = x.mean(axis=0)
        std_raw = x.std(axis=0)

        # force the result to be at least a 1D array so we can modify its elements
        self.scaler_std_ = np.atleast_1d(std_raw)

        # prevent division by zero for constant features
        self.scaler_std_[self.scaler_std_ == 0] = 1.0


    def _transform_features(self, X: np.ndarray) -> np.ndarray:
        if self.scaler_mean_ is None or self.scaler_std_ is None:
            raise RuntimeError("Scaler has not been fit yet.")
        return ((X - self.scaler_mean_) / self.scaler_std_).astype(np.float32)

    def predict(self, x: pd.DataFrame = None) -> np.ndarray:
        """
        Predict using parameter x or full dataset (if final model), or x_test (if testing model).
        Returns mean prediction only.
        """
        if self.model is None:
            raise RuntimeError("MLPRegression model has not been trained yet.")

        if (self._test_size > 0 or self._pre_split) and x is None:
            X_eval = self._x["test"]
        else:
            X_eval = self._x["full"] if x is None else x

        X_array = np.asarray(X_eval, dtype=np.float32)
        X_scaled = self._transform_features(X_array)
        raw = self.model.predict(X_scaled, verbose=0)

        num_targets = len(self.targets)

        # Grab means and inverse scale them
        mu_scaled = raw[:, :num_targets]
        self._predictions = (mu_scaled * self.y_std_) + self.y_mean_

        # If we predicted on the test set, update scores
        if (self._test_size > 0 or self._pre_split) and x is None:
            self._score()

        return self._predictions

    def predict_mean_variance(self, X):
        """Return mean and variance for deep ensemble use."""
        if self.model is None:
            raise RuntimeError("MLPRegression model has not been trained yet.")

        X_array = np.asarray(X, dtype=np.float32)
        X_scaled = self._transform_features(X_array)
        raw = self.model.predict(X_scaled, verbose=0)

        num_targets = len(self.targets)

        # Slice means and log variances
        mu_scaled = raw[:, :num_targets]
        log_var_scaled = np.clip(
            raw[:, num_targets:],
            self._tf_params.get("log_var_min", -10.0),
            self._tf_params.get("log_var_max", 10.0),
        )
        var_scaled = np.exp(log_var_scaled)

        # Inverse transform to real-world values
        mu_real = (mu_scaled * self.y_std_) + self.y_mean_
        var_real = var_scaled * (self.y_std_ ** 2)

        return mu_real, var_real
