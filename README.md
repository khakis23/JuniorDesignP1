# AutoML Implementation
### Junior Design Spring 2026
### Authors: Tyler Black, Reed Delp, Ayiana McCabe

## Purpose
The purpose of this project is to implement an AutoML structure to build a
foundation for different machine learning projects. The architecture supports
various different models and customizable features. However, the content of
this README will mainly focus on a specific uncertainty quantification (UQ) 
implementation.


## Architecture

The architecture is built around a high-level interface class called
`ModelMaker`. `ModelMaker` has an intuative syntax that gives the user full
access to controlling the implimented models while abstracting away unnecessary
complications and proceedures. See how to use the `ModelMaker` class in the _usage_
section. The basic architecture is shown below. When the user instructs `ModelMaker`
to train a model, it is passed in the `Trainer` class, which handles the entire
training proceedure for any implimented `IModel`. The trained model(s) are returned
to `ModelMaker` before they are passed into the 'ModelEval' class, which blindly
evaluates the best models based on their type and available scores. The 
`ModelEval` also handles optional printing, plotting, and saving of the results.
Currently, an `IModel` can be implimented using any _TensorFlow_ or _SKLearn_ model.

![Architecure](/readme_assets/arch.png)


## Usage

`ModelMaker` is the primary API wrapper for model training, evaluation, grid searches, and deep ensemble generation. 

### General Workflow

1. **Instantiate** `ModelMaker()`.
```python
mm = ModelMaker()
```
2. **Define data** from DataFrame or built-in dataset string added to `Models/src/factories.py`
3. **Select features** by passing in a list of lists of feature names, or by calling `util` function: `feature_combo`.
```python
features = [
    ["A", "B", "C", "D"],
    ["A", "B", "C"],
    ["A", "D"],
]
```
3. **Define parameters** via a dictionary matching the underlying model's `kwargs`.
```python
params = {
        "tts": [0.2],
        "folds": [4],
        "early_stopping": [1],
        "hidden_layer_sizes": [
            (32, 16),
            (64, 32),
        ],
        "activation": ["relu", "elu", "swish"]
}
```
4. **Search & Evaluate:** Call `train_and_eval()` to execute a grid or random search.
```python
mm.train_and_eval(
    "MLPRegression",
    "CHF",
    features_list=features,
    params=params,
    autosave="CV R2",   # (optional) automatically save the best model specified 
    plot_func=plot_actual_vs_pred,  # (optional) plotting function
    random_state=42,
    random_search=300,   # (optional) leave blank for grid search
)
```
5. **Save:** (Optional) Call `save_best()` or use the `autosave` parameter to store top models in memory.
6. **Ensemble:** Call `train_eval_deep_ens()` using a saved model, the active best model, or manual configurations.
```python
mm.train_eval_deep_ens(
    "CHF",
    plot_func=deep_ensemble_parity_plot,  # (optional) plotting function
    n_models=10,
    random_state=42,
)
```
7. **Access:** Retrieve saved models via `ModelMaker.best` or `get_saved_model()` for plotting and predictions.
```python
model = mm.get_saved_model(0)
model.predict(new_data)
```

### Core Methods

#### `train_and_eval()`
Executes a grid or random search across hyperparameter/feature combinations, ranks them by standard metrics, and displays the top performers.
* **`model_name`**: Name of the model to train (e.g., `"MLPRegression"`).
* **`dataset`**: DataFrame or string name of a built-in dataset.
* **`features_list`**: List of feature lists to test (e.g., `[["f1", "f2"], ["f1"]]`).
* **`params`**: Dictionary of parameters to grid search (e.g., `{"alpha": [0.1, 0.2]}`).
* **`target_cols`**: List of target columns.
* **`random_search`**: (Optional) Int limit for random sampling of the grid.
* **`autosave`**: (Optional) String of the metric (e.g., `"R2"`) to automatically save the best-scoring model.

#### `train_eval_deep_ens()`
Trains and evaluates a Deep Ensemble. Automatically figures out how to build the base architecture based on your inputs:
* **Method 1 (Saved Index)**: Pass `best_idx` to use a previously saved model.
* **Method 2 (Implicit Best)**: Pass nothing for model parameters; it will automatically use `self.best`.
* **Method 3 (Custom)**: Manually pass `base_model`, `features`, `target_cols`, and `params`.
* **`n_models`**: Number of ensemble members to train.

#### State Management
After a model has been trained, it can be saved to persistent memory for later use, by accessing `ModelMaker.best`, which
directly accesses the instance of the model.
* **`save_best(idx_or_model)`**: Manually saves a model to persistent memory (by evaluation index or direct object).
* **`get_saved_model(idx)`**: Retrieves a previously saved model by its chronological save index.
* **`clear_evaluator()`**: Wipes the internal evaluation state (useful between isolated training runs).


## Environment Setup

Environment setup for this project varies between systems. The following setup
is tested on **ARM-based MacOS** and **headless Ubuntu 24.04 equipped with an ASUS GeForce RTX 5070** using the latest stable 
NVIDIA drivers as of March 2026. **All environments REQUIRE Python 3.11.**

For similar linux-based systems:
```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

For Apple Silicon, getting tensorflow to run on the GPU can require a bit
of extra effort. Try the following manual installations:
```bash
python3.11.5 -m venv .venv  # or might also be fine >> python3.11 -m venv .venv
source .venv/bin/activate
pip install tensorflow-macos==2.15.0 tensorflow-metal==1.1.0
pip install xarray==2026.4.0 opencv-python==4.13.0.92 skglm==0.5 tinycss2==1.4.0 pytest==9.0.3 matplotlib==3.10.8
```

A Windows system was not tested.

---

# pyMAISE, Uncertainty Quantification, & Deep Ensembles

## Deep Ensembles

Deep Ensembles are a practical and widely used approach for estimating uncertainty in neural network predictions. Instead of relying on a single model, the method trains several neural networks with the identical architectures but different random starting conditions. Because each model learns slightly different patterns from the data, their predictions vary, and this variation can be used to estimate how confident the model is in its results.

In this project, we implement a Deep Ensemble model and use it to quantify predictive uncertainty using statistical measures such as the mean, variance, and 95% confidence interval. The model is then applied to a selected pyMAISE benchmark problem to evaluate its prediction accuracy and its ability to solve uncertainty in the outputs.


### Implementation

Like all other models, the Deep Ensemble extends the `IModel` base class.
THis allows the Deep Ensemble to act like any other single model. The Deep Ensemble
has a few special cases that are mostly hardcoded in either `Trainer` or `ModelMaker`.
Other cases are dynamically handles by `kwargs`, just like any other model.

---

#### `train_and_fit()` 
The Deep Ensemble class overrides `train_and_fit()`, which is where each base-model
is trained and the ensemble is assembled. A random state is assigned to each model
using the `random_state` parameter plus the member index.
```python
# Error handling and visualization removed for readability
def train_and_fit(self, tts: float = 0.0, random_state=42, **kwargs):
    # Base class splitting
    super().train_and_fit(tts, random_state)

    # Allow passing class via kwargs or use the one from __init__
    self.base_model_class = kwargs.pop("base_model_class", self.base_model_class)
    self.n_models = kwargs.pop("n_models", self.n_models)

    self.models = []
    self.member_scores = []

    # Store ensemble configuration for logging
    self._parameters = {
        "base_model_class": getattr(self.base_model_class, "__name__", str(self.base_model_class)),
        "n_models": self.n_models,
        **kwargs,
    }

    for i in range(self.n_models):
        # Advance seed to ensure diversity in initializations and training shuffles
        member_seed = None if random_state is None else random_state + i

        # Instantiate a fresh instance of the base model
        # Pass targets and data_test to support multi-output and pre-split
        model = self.base_model_class(self.features, self.targets, self.data, self.data_test)

        # Pass the split ratio and all keras kwargs down to the member model
        model.train_and_fit(tts=tts, random_state=member_seed, **dict(kwargs))
        self.models.append(model)

        # gather scores if the member evaluated on test data
        score = model.get_scores()
        self.member_scores.append(score)
```
---
#### `predict()`
The `predict()` method implements the main uncertainty quantification logic
for the Deep Ensemble. The method iterates through each base model and calls
the **required** `predict_mean_variance()` method to obtain the aleatric 
variance. The method stores the collection of vrainces and outputs in two lists,
which are used to calculate the ensemble mean and both the aleatoric and epistemic
variances.

```python
# Error handling removed for readability
def predict(self, x: pd.DataFrame = None, cv_folds: int = 0) -> np.ndarray:
    # Determine which x to use
    if (self._test_size > 0 or self._pre_split) and x is None:
        x_eval = self._x["test"]
    else:
        x_eval = self._x["full"] if x is None else x

    mus = []     # predictions
    vars_ = []   # variances

    for idx, model in enumerate(self.models):
        # Get the predicted mean and variance from single base model
        mu_m, var_m = model.predict_mean_variance(x_eval)

        # Removed reshape(-1) to keep the multi-output shape (samples, targets)
        mu_m = np.asarray(mu_m)
        var_m = np.asarray(var_m)

        mus.append(mu_m)
        vars_.append(var_m)

    # Use np.stack to create a 3D array: (n_models, n_samples, n_targets)
    mus_stacked = np.stack(mus, axis=0)
    vars_stacked = np.stack(vars_, axis=0)

    self.member_means = mus_stacked
    self.member_vars = vars_stacked

    # Deep Ensemble Logic:
    # np.mean and np.var on axis=0 collapses the models dimension,
    # leaving us with (n_samples, n_targets) arrays.
    self.mean_prediction = np.mean(mus_stacked, axis=0)
    self.epistemic_var = np.var(mus_stacked, axis=0)
    self.aleatoric_var = np.mean(vars_stacked, axis=0)

    self.total_var = self.epistemic_var + self.aleatoric_var
    self.prediction_std = np.sqrt(self.total_var)

    self._predictions = self.mean_prediction

    # Update overall scores if evaluating on the test set
    if (self._test_size > 0 or self._pre_split) and x is None:
        self._score()

    return self.mean_prediction
```


## pyMAISE Example Datasets

### Critical Heat Flux (CHF) Dataset

In a nuclear reactor core, CHF refers to the limit at which wall heat transfer significantly decreases. This measurement
becomes important as significant wall temperatures are resultant of unchecked CHF. This is concurrent with wall oxidation
and potentially fuel rod failure.

As noted from the Nuclear Energy Agency: "CHF is challenging to predict accurately due to the complexities of the involved
phenomena". Relative to past projects, the Deep Ensemble is a good fit for this dataset.

For the purposes of this project, the CHF dataset consists of 2500 samples of measurements taken from the PyMAISE
benchmark CHF dataset. Each sample contains measurements for pressure, test section diameter, mass flux, inlet
temperature, outlet equilibrium quality, and CHF; 6 inputs to 1 output. Thus, the deep ensemble's goal is to predict
CHF and return uncertainty given the six inputs specified previously.

### CHF Dataset Results
![CHF Dataset Results](/readme_assets/DE_CHF.png)

### MIT Reactor Dataset

A nuclear reactors control blades or rods serve a critical role of managing electrical power output of the station itself.
Measurement devices used to record power readings must be regularly tested to avoid premature dropping (in the case of BWRs:
 scramming) or heightening of control elements.

In the case of machine learning models, they could supplement physical sensors. For this, a reliable and well-trained model will
return a power prediction independent of actual power readings. From there, any dissonance between measurements are anomalies 
and could indicate drifting of physical instruments.

A Deep Ensemble's inherent uncertainty quantification serves as a check against false flags in anomalous model to actual power
readings. Should any unfamiliar control rod configurations arise, a Deep Ensemble will make its unfamiliarity known as a
failsafe.

The reactor dataset contains 1000 samples, each with 6 inputs and 22 outputs. The Deep Ensemble is trained to predict power
output of the 22 fuel elements given the height of each of the 6 control blades.

### MIT Reactor Dataset Results

### Boiling Water Reactor (BWR) Micro Core Dataset

Given a set of parameters for the physical geometry of fuel elements and cooling conditions, the power density of a
reactor across its fuel elements can be predicted. In relation with Deep Ensembles, the model can be applied to measure
the efficacy of given fuel rod arrangements when planning and constructing a nuclear reactor (micro-reactor in this case).

The BWR Dataset contains 2000 samples, each with 9 inputs and 3 outputs. The inputs can be divided into 3 sections:
vertical fuel distribution, cooling conditions, and control element geometry/arrangment. Inputs have 2 sections: stability,
and localized power density in a 2D slice and 3D mapping.

### BWR Micro Core Dataset Results


## Discussion & Conclusion

TODO 
