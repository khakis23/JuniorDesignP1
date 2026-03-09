# Model Hyperparameter Documentation

## 1. Ridge Regression (`RidgeCV`)
Ridge regression is a linear model that applies L2 regularization to prevent overfitting by penalizing large coefficients. Because we use `RidgeCV`, it automatically performs cross-validation to find the best `alpha` from a list you provide.

| Parameter | Description | Typical Values / Default | Best Value |
| :--- | :--- | :--- | :--- |
| `alphas` | An array of regularization strengths to test. Higher values specify stronger regularization. | `(0.1, 1.0, 10.0)` |  |
| `cv` | Determines the cross-validation splitting strategy. | `None` (Leave-One-Out), `5`, `10` |  |
| `fit_intercept` | Whether to calculate the intercept for this model. | `True`, `False` |  |
| `scoring` | A string indicating the metric used to evaluate the cross-validation performance. | `None` (uses $R^2$), `'neg_mean_squared_error'` |  |

---

## 2. MCP Regression (`skglm.MCPRegression`)
Minimax Concave Penalty (MCP) regression is a sparse linear model. It acts similarly to Lasso (L1) but applies a non-convex penalty that relaxes the regularization on larger coefficients, reducing bias.

| Parameter | Description | Typical Values / Default | Best Value |
| :--- | :--- | :--- | :--- |
| `alpha` | Constant that multiplies the penalty term. Determines the overall strength of the regularization. | `0.001`, `0.01`, `0.1`, `1.0` |  |
| `gamma` | The MCP-specific parameter that controls the concavity of the penalty. **Must be strictly > 1**. Closer to 1 mimics Lasso; higher values approach unpenalized OLS for large coefficients. | `1.5`, `3.0`, `5.0` |  |
| `max_iter` | The maximum number of iterations for the optimization solver to converge. | `100`, `1000`, `2000` |  |
| `tol` | Tolerance for the optimization. The solver stops when updates fall below this value. | `1e-4` |  |
| `fit_intercept`| Whether to calculate the intercept for this model. | `True`, `False` |  |

---

## 3. LSTM Regression (Custom Keras Architecture)
A recurrent neural network architecture designed to recognize patterns over chronological sequences. 

*Note: `lookback` is passed during instantiation (`__init__`), while the rest are passed to `train_and_fit()` via `**kwargs`.*

| Parameter | Description | Typical Values / Default | Best Value |
| :--- | :--- | :--- | :--- |
| `lookback` | How many historical hours/timesteps the model looks at to predict the target. | `12`, `24`, `48` |  |
| `epochs` | The maximum number of passes through the training dataset. (EarlyStopping will likely halt training before this max is reached). | `100`, `300`, `500` |  |
| `batch_size` | Number of sequence samples processed before the model updates its internal weights. | `16`, `32`, `64`, `128` |  |
| `lstm_units_1` | The number of internal memory units in the first (Bidirectional) LSTM layer. | `64`, `128`, `256` |  |
| `lstm_units_2` | The number of internal memory units in the second standard LSTM layer. | `32`, `64`, `128` |  |
| `dense_units` | The number of neurons in the final fully-connected feed-forward layer before the output. | `16`, `32`, `64` |  |
| `dropout_rate` | The fraction of neurons randomly disabled during each training step to prevent overfitting. | `0.1`, `0.2`, `0.3`, `0.4` |  |
| `validation_split` | The fraction of training data reserved to monitor `val_loss` for EarlyStopping and Learning Rate reduction. | `0.1`, `0.15`, `0.2` |  |