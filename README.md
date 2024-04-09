# Reguarlizing the Forward Pass

[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

In certain applied microeconomic settings, it's typical to view one's dataset as the realization of a stratified cluster randomized control trial - treatment is assigned at the cluster level and controls vary at both the individual and cluster level. 
<div align="center">
  <img src="https://github.com/pharringtonp19/rfp/assets/55798098/6a4f1cdc-0b09-4e76-b75a-8d7c1e77d7e0"" width="40%" height="auto">
</div>

Locally, this makes it more likely that observation will be from the same cluster (see below!) which can increase the variance for estimators which don't account for the clustered nature of the data.
<div align="center">
  <img src="https://github.com/pharringtonp19/rfp/assets/55798098/831c9f46-7e77-411e-8f03-c988e9dd1e51" width="40%" height="auto">
</div>

 We introduce a framework for partialling out nonparametric cluster effects in a way that generalizes least squares and is inherently compositional even under regularization.


