## Overview
The [standard training](https://github.com/pharringtonp19/rfp/blob/main/notebooks-scripts/standard_training.ipynb) notebook estimates a conditional expectation function $\mathbb{E}[Y \vert W]$ via a feed-forward neural network. We can use this notebook to construct estimates of the LATE effect by setting $W := (X, Z)$ and taking $Y$ to be the outcome and treatment variable for separate runs.

$$ \theta := \int \frac{\mathbb{E}[Y\vert X,Z=1] - \mathbb{E}[Y\vert X,Z=0]}{\mathbb{E}[D\vert X,Z=1] - \mathbb{E}[D\vert X,Z=0] } d\mathbb{P}$$

The [rfp training](https://github.com/pharringtonp19/rfp/blob/main/notebooks-scripts/rfp_training.ipynb) notebook estimates a conditional expectation function $\mathbb{E}[Y \vert W]$ by fitting a feed-forward neural network using regularized bi-level gradient descent.


## **Implementation Details**

- We allow observational weighting via `sample_weights`
- Hyperparameters are specified in the **Parameterize Notebook** cell block


## Potential Issues

#### Binary Targets
With binary targets, nans can occur during the training run if the predicted value is too close to 0 or 1. We therefore suggest altering the feature normalization procedure and using the following function to plot a histogram of the initial values 

```python
def hist_predictions(key):
    params = ModelParams.init_fn(key, mlp, features)
    yhat = model.fwd_pass(params, XZ_normalized)
    plt.hist(yhat.reshape(-1,), bins=50)
    plt.show()
```
