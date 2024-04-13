#!/bin/bash

# Name of the virtual environment
ENV_NAME="rfp_env"

# Create virtual environment
python3 -m venv $ENV_NAME

# Activate the virtual environment
source $ENV_NAME/bin/activate

# Install any necessary Python packages
pip install --upgrade "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
pip install optax, flax
pip install git+https://github.com/pharringtonp19/rfp.git
pip install matplotlib
