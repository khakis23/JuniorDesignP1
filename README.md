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

The architecture is built around an easy high-level interface class called
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

TODO


## Dependencies

TODO


# pyMAISE and Uncertainty Quantification

## Deep Ensembles

Deep Ensembles are a practical and widely used approach for estimating uncertainty in neural network predictions. Instead of relying on a single model, the method trains several neural networks with the identical architectures but different random starting conditions. Because each model learns slightly different patterns from the data, their predictions vary, and this variation can be used to estimate how confident the model is in its results.

In this project, we implement a Deep Ensemble model and use it to quantify predictive uncertainty using statistical measures such as the mean, variance, and 95% confidence interval. The model is then applied to a selected pyMAISE benchmark problem to evaluate its prediction accuracy and its ability to solve uncertainty in the outputs.


## Uncertainty Quantification

TODO


## pyMAISE Datasets

### Dataset 1

TODO DESCRIPTION  — REED

### Dataset 1 Results

TODO RESULTS AND PLOTS — We can all do one of these? for each dataset? short ans sweet

### TODO THE REST OF THE DATASETS...


## Discussion & Conclusion

TODO 
