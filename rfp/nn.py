from typing import Sequence, Callable
import jax
from flax import linen as nn
import jax.numpy as jnp
from flax.core import unfreeze

class MLP(nn.Module):
    nodes: Sequence[int]
    activation: Callable = nn.relu
    final_activation: Callable = nn.sigmoid

    @nn.compact
    def __call__(self, x):
        for feat in self.nodes:
            x = self.activation(nn.Dense(feat)(x))
        return x

    def init_fn(self, key, features, verbose=False):
        """Initialize Parameters"""
        params = unfreeze(self.init(key, jnp.ones((features,))))["params"]
        if verbose:
            print(jax.tree_util.tree_map(lambda x: x.shape, params))  # Checking output shapes
        return params

    def fwd_pass(self, params, x):
        """The Forward Pass"""
        return self.apply({"params": params}, x)


    def embellished_fwd_pass(self, params, x):
        return self.fwd_pass(params, x), 0.0
    
    def predict(self, full_params, x):
        phiX = self.fwd_pass(full_params.body, x)
        yhat = self.final_activation(self.final_layer(phiX))
        return yhat

    @staticmethod
    def final_layer(full_params, x):
        """Apply a final layer of the model"""
        return x @ full_params.head + full_params.bias
