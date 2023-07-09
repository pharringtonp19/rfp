from typing import Callable, Sequence
import flax.linen as nn
import jax
import jax.numpy as jnp
from flax.core import unfreeze


class MLP(nn.Module):
    nodes: Sequence[int]
    activation: Callable = nn.relu

    @nn.compact
    def __call__(self, x):
        for feat in self.features:
            x = self.activation(nn.Dense(feat)(x))
        return x

    def fwd_pass(self, params, x):
        """The Forward Pass"""
        return self.apply({"params": params}, x)

    def init_fn(self, key, features, verbose=False):
        """Initialize Parameters"""
        params = unfreeze(self.init(key, jnp.ones((features,))))["params"]
        if verbose:
            print(jax.tree_util.tree_map(lambda x: x.shape, params)) # Checking output shapes
        return params

    def embellished_fwd_pass(self, params, x):
        return self.fwd_pass(params, x), 0.0
