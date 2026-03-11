# Photovoltaic Energy Forcasting Using Machine Learning
### Authors: Tyler Black, Reed Delp, Ayaina McCabe

---

## Model Parameters

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
| **Test Size** |  | 0.2        |


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
| **Test Size** |  | 0.2        |


## Model Score Comparison

| Metric           | Random Forest | LSTM Regression |
|:-----------------|:--------------| :--- |
| **$R^2$**        | 0.917         | 0.935 |
| **CV $R^2$**          | 0.925         | - |
| **RMSE**         | 181.009       | 137.920 |
| **RMSE Clamped** | 181.165       | 137.488 |
| **MAE**          |  89.919              | 75.055 |
| **CI**           | 0.905, 0.927  | 0.925, 0.944 |
