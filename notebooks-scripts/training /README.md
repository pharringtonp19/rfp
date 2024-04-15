## Overview
The [standard training](https://github.com/pharringtonp19/rfp/blob/main/notebooks-scripts/standard_training.ipynb) notebook estimates a conditional expectation function $\mathbb{E}[Y \vert W]$ via a feed-forward neural network. We can use this notebook to construct estimates of the treatment effects by setting $W := (X, D)$ or $W := (X, Z)$ and taking $Y$ to be the outcome and treatment variable for separate runs.

The [rfp training](https://github.com/pharringtonp19/rfp/blob/main/notebooks-scripts/rfp_training.ipynb) notebook estimates a conditional expectation function $\mathbb{E}[Y \vert W]$ by fitting a feed-forward neural network using regularized bi-level gradient descent.

## **Treatment Heterogeneity**
By fitting nonparametric conditional expectation functions we can estimate treatment heterogeneity across some variable $K$ by average over the nuisance dimensions $X$.

$$\mathbb{E}[Y_i \vert D_i = 1, K_i] = \int \mathbb{E}[Y_i \vert D_i = 1, X_i, K_i] \mathbb{P}_{X_i \vert K_i}$$



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
