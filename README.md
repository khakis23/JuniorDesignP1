# Photovoltaic Energy Forcasting Using Machine Learning
### Authors: Tyler Black, Reed Delp, Ayaina McCabe

---

## Model Parameters

### Ridge Regression

| Parameter | Function | Best Value |
| :--- | :--- | :--- |
| **Alpha** | The regularization strength that penalizes large coefficients to prevent the model from overfitting to noisy data. | 0.305 |
| **Test Size** | The proportion of the dataset held back to evaluate the final model's performance. | 0.2 |

### MLP Regression

| Parameter | Function | Best Value |
| :--- | :--- | :--- |
| **Hidden Layers** | The number of neurons inside each sequential layer of the neural network. | 256, 128, 64, 32 |
| **Hidden Activation** | The mathematical function used by the hidden layers to learn non-linear patterns. | relu |
| **Output Activation** | The function used at the final layer to produce the actual predicted value. | identity |
| **Alpha** | The L2 regularization penalty applied to the network's weights to keep the model from overfitting. | 0.01 |
| **Learning Rate Init** | The starting step size the model uses to update its internal weights during training. | 0.005 |
| **Max Iter** | The absolute maximum number of training epochs allowed if early stopping doesn't trigger. | 800 |
| **Early Stopping** | A toggle (1 or 0) that stops training automatically when the model stops improving on the validation set. | 1 |
| **Batch Size** | The number of data samples processed at once before the model updates its weights. | 64 |
| **Solver** | The specific optimization algorithm used to minimize the error during training. | adam |
| **Validation Fraction** | The percentage of training data held out strictly for early stopping checks. | 0.1 |
| **Test Size** | The proportion of the dataset held back to evaluate the final model's performance. | 0.2 |

### Random Forest

| Parameter             | Function | Best Value |
|:----------------------| :--- |:-----------|
| **Estimators** | The total number of decision trees built to vote on the final prediction. | 200        |
| **Max Depth** | The maximum number of splits allowed per tree so it learns patterns instead of just memorizing the training data. | 15         |
| **Min Samples Split** | The minimum amount of data points needed in a node before it's allowed to split again. | 2          |
| **Min Samples Leaf** | The minimum amount of data points that have to end up at the final tip of a branch. | 2          |
| **Max Features** | The fraction of your columns the model looks at when deciding how to split the data. | 0.33       |
| **Bootstrap** | A simple yes/no (1 or 0) on whether the model randomly samples data with replacement. | 1          |
| **Features Seen** | The total number of input variables the model used. | 9          |
| **Test Size** | The proportion of the dataset held back to evaluate the final model's performance. | 0.2        |

### Gradient Boosting

| Parameter | Function | Best Value |
| :--- | :--- | :--- |
| **Estimators** | The total number of sequential trees built to correct the errors of previous trees. | 100 |
| **Learning Rate** | The step size shrinking the contribution of each new tree to prevent overfitting. | 0.1 |
| **Max Depth** | The maximum number of splits allowed per tree so it learns patterns instead of just memorizing the training data. | 5 |
| **Loss** | The specific error metric the model tries to minimize while adding new trees. | squared_error |
| **Subsample** | The fraction of training data used to fit each individual tree. | 1.0 |
| **Min Samples Split** | The minimum amount of data points needed in a node before it's allowed to split again. | 2 |
| **Min Samples Leaf** | The minimum amount of data points that have to end up at the final tip of a branch. | 1 |
| **Test Size** | The proportion of the dataset held back to evaluate the final model's performance. | 0.2 |

### LSTM Regression

| Parameter | Function | Best Value |
| :--- | :--- |:-----------|
| **Lookback** | How many past time steps the model looks at to predict the next one. | 48         |
| **Epochs** | How many times the model runs through the entire training dataset. | 500        |
| **Batch Size** | How many rows of data the model processes at a time before updating itself. | 32         |
| **LSTM 1** | The number of memory units in the first sequence-processing layer. | 128        |
| **LSTM 2** | The number of memory units in the second sequence-processing layer. | 64         |
| **LSTM 3** | The number of memory units in the third sequence-processing layer. | 32         |
| **Dropout** | The percentage of nodes randomly turned off during training so the model doesn't overfit. | 0.3        |
| **Dense Units** | The number of nodes in the standard layer right before the final output. | 16         |
| **Validation Split** | The chunk of data held back during training to check performance after every single epoch. | 0.1        |
| **Test Size** | The proportion of the dataset held back to evaluate the final model's performance. | 0.2        |

---

## Model Score Comparison

| Metric | Ridge | MLP Regression | Random Forest | Gradient Boosting | LSTM Regression |
|:---|:---|:---|:---|:---|:---|
| **$R^2$** | 0.843 | 0.912 | 0.917 | 0.913 | 0.935 |
| **CV $R^2$** | - | 0.910 | 0.925 | 0.918 | - |
| **RMSE** | 248.641 | 185.720 | 181.009 | 184.841 | 137.920 |
| **RMSE Clamped** | 225.334 | 185.537 | 181.165 | 184.981 | 137.488 |
| **MAE** | 184.941 | 95.373 | 89.919 | 93.530 | 75.055 |
| **CI** | 0.830, 0.854 | 0.900, 0.924 | 0.905, 0.927 | 0.900, 0.925 | 0.925, 0.944 |