import numpy as np
import tensorflow as tf

from Data.ModelData.ModelData import ModelData
from Models.src.IModel import IModel
#from util.util import plot_daily_solar_output


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
        - predict(X) returns mean only, so the rest of the project can still work.
        - predict_mean_variance(X) returns (mu, var) for deep ensemble use.
    """

    def __init__(self, features: list[str], data: ModelData):
        super().__init__(features, data)
        self.model: tf.keras.Model | None = None
        self.history = None
        self.scaler_mean_ = None
        self.scaler_std_ = None
        self._tf_params = {}

    def train_and_fit(self, random_state=42, **kwargs):
        tts = kwargs.pop("tts", 0.0)
        super().train_and_fit(tts, random_state)

        tf.keras.utils.set_random_seed(random_state)

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

        if kwargs:
            unknown = ", ".join(sorted(kwargs.keys()))
            raise TypeError(f"Unexpected MLPRegression kwargs: {unknown}")

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
            "output_type": "mean_variance",
        }

        x_key = "train" if self._test_size else "full"
        y_key = x_key

        X_fit = np.asarray(self._X[x_key], dtype=np.float32)
        y_fit = np.asarray(self._y[y_key], dtype=np.float32).reshape(-1, 1)

        self._fit_scaler(X_fit)
        X_fit_scaled = self._transform_features(X_fit)

        self.model = self._build_model(
            input_dim=X_fit_scaled.shape[1],
            hidden_layer_sizes=hidden_layer_sizes,
            activation=activation,
            optimizer_name=optimizer_name,
            learning_rate=learning_rate,
            l2_alpha=l2_alpha,
        )

        callbacks = []
        if early_stopping:
            callbacks.append(
                tf.keras.callbacks.EarlyStopping(
                    monitor="val_loss",
                    patience=patience,
                    min_delta=min_delta,
                    restore_best_weights=True,
                )
            )

        fit_kwargs = {
            "x": X_fit_scaled,
            "y": y_fit,
            "epochs": epochs,
            "batch_size": batch_size,
            "verbose": verbose,
            "callbacks": callbacks,
        }

        if validation_split and len(X_fit_scaled) > 1:
            fit_kwargs["validation_split"] = validation_split

        self.history = self.model.fit(**fit_kwargs)

        # Let the shared IModel flow build predictions/scores the same way as other models.
        self._fit()

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

        # Two outputs:
        #   [:, 0] = mu
        #   [:, 1] = log_var
        model.add(tf.keras.layers.Dense(2, activation="linear"))

        optimizer = self._make_optimizer(optimizer_name, learning_rate)
        model.compile(optimizer=optimizer, loss=self.gaussian_nll)
        return model

    def gaussian_nll(self, y_true, y_pred):
        mu = y_pred[:, 0:1]
        log_var = y_pred[:, 1:2]

        log_var = tf.clip_by_value(
            log_var,
            self._tf_params.get("log_var_min", -10.0),
            self._tf_params.get("log_var_max", 10.0),
        )

        precision = tf.exp(-log_var)
        return tf.reduce_mean(0.5 * (log_var + tf.square(y_true - mu) * precision))

    @staticmethod
    def _make_optimizer(name: str, learning_rate: float):
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

    def _fit_scaler(self, X: np.ndarray):
        self.scaler_mean_ = X.mean(axis=0)
        self.scaler_std_ = X.std(axis=0)
        self.scaler_std_[self.scaler_std_ == 0] = 1.0

    def _transform_features(self, X: np.ndarray) -> np.ndarray:
        if self.scaler_mean_ is None or self.scaler_std_ is None:
            raise RuntimeError("Scaler has not been fit yet.")
        return ((X - self.scaler_mean_) / self.scaler_std_).astype(np.float32)

    def predict(self, X):
        """Return mean prediction only."""
        if self.model is None:
            raise RuntimeError("MLPRegression model has not been trained yet.")

        X_array = np.asarray(X, dtype=np.float32)
        X_scaled = self._transform_features(X_array)
        raw = self.model.predict(X_scaled, verbose=0)
        mu = raw[:, 0]
        return mu.reshape(-1)

    def predict_mean_variance(self, X):
        """Return mean and variance for deep ensemble use."""
        if self.model is None:
            raise RuntimeError("MLPRegression model has not been trained yet.")

        X_array = np.asarray(X, dtype=np.float32)
        X_scaled = self._transform_features(X_array)
        raw = self.model.predict(X_scaled, verbose=0)

        mu = raw[:, 0]
        log_var = np.clip(
            raw[:, 1],
            self._tf_params.get("log_var_min", -10.0),
            self._tf_params.get("log_var_max", 10.0),
        )
        var = np.exp(log_var)
        return mu.reshape(-1), var.reshape(-1)

    def _score(self, folds: int = 5):
        super()._score()

        history_loss = self.history.history.get("loss", []) if self.history else []
        history_val_loss = self.history.history.get("val_loss", []) if self.history else []

        self._parameters = {
            "Test Size": self._test_size,
            "Epochs Requested": self._tf_params.get("epochs"),
            "Epochs Trained": len(history_loss),
            "Final Loss": float(history_loss[-1]) if history_loss else None,
            "Final Validation Loss": float(history_val_loss[-1]) if history_val_loss else None,
            "Hidden Layers": self._tf_params.get("hidden_layer_sizes"),
            "Hidden Activation": self._tf_params.get("activation"),
            "Optimizer": self._tf_params.get("optimizer"),
            "Learning Rate": self._tf_params.get("learning_rate"),
            "Batch Size": self._tf_params.get("batch_size"),
            "Validation Split": self._tf_params.get("validation_split"),
            "Early Stopping": self._tf_params.get("early_stopping"),
            "L2 Alpha": self._tf_params.get("l2_alpha"),
            "Output Type": self._tf_params.get("output_type"),
            "Log Var Min": self._tf_params.get("log_var_min"),
            "Log Var Max": self._tf_params.get("log_var_max"),
        }

    def get_parameters(self) -> dict:
        return dict(self._tf_params)

    def plot(self):
        title = "MLP Regression (Mean-Variance)"

        if self._test_size:
            ci = self._scores.get("CI", (0.0, 0.0))
            ci_display = f"[{round(ci[0], 2)} : {round(ci[1], 2)}]" if isinstance(ci, (list, tuple)) else str(round(ci, 2))

            display_features = {
                "R2": round(self._scores["R2"], 3),
                "CI": ci_display,
                "RMSE": round(self._scores["RMSE Clamped"], 2),
                "Epochs": self._parameters.get("Epochs Trained"),
                "Layers": self._parameters.get("Hidden Layers"),
            }

            plot_daily_solar_output(
                self._c_predictions,
                self._y["test"],
                title,
                display_features,
            )
        else:
            plot_daily_solar_output(
                self._c_predictions,
                self._y["full"],
                title,
            )
