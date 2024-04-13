The [standard training](https://github.com/pharringtonp19/rfp/blob/main/notebooks-scripts/standard_training.ipynb) notebook estimates a conditional expectation function $\mathbb{E}[Y \vert W]$ via a feed-forward neural network. We can use this notebook to construct estimates of the LATE effect by setting $W := (X, Z)$ and taking $Y$ to be the outcome and treatment variable for separate runs.

$$ \theta := \frac{\mathbb{E}[Y\vert X,Z=1] - \mathbb{E}[Y\vert X,Z=0]}{\mathbb{E}[D\vert X,Z=1] - \mathbb{E}[D\vert X,Z=0] } $$

We allow observational weighting via `sample_weights`
