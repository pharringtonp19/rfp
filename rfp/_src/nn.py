"""

Examples:
    >>> import jax 
    >>> import jax.numpy as jnp 
    >>> import flax.linen as nn
    >>> from typing import Sequence
    >>> from flax.core import unfreeze

    >>> n, d = 100, 10 # (observations, features)
    >>> mlp = MLP([32, 1], activation=nn.relu) 
    >>> xs = jax.random.normal(jax.random.PRNGKey(0), (n, d))
    >>> params = mlp.init_fn(jax.random.PRNGKey(1), d)
    >>> yhat = mlp.fwd_pass(params, xs)
"""
from typing import Callable, Sequence

import flax.linen as nn
import jax
import jax.numpy as jnp
from flax.core import unfreeze

from rfp._src.types import Array


class MLP(nn.Module):
    features: Sequence[int]
    activation: Callable = nn.relu

    @nn.compact
    def __call__(self, x: Array):
        for feat in self.features[:-1]:
            x = self.activation(nn.Dense(feat)(x))
        x = nn.Dense(self.features[-1])(x)
        return x

    def fwd_pass(self, params, x):
        """The Forward Pass"""
        return self.apply({"params": params}, x)

    def init_fn(self, key, features):
        """Initialize Parameters"""
        params = unfreeze(self.init(key, jnp.ones((1, features))))["params"]
        return params

    def embellished_fwd_pass(self, params, x):
        return self.fwd_pass(params, x), 0.0


if __name__ == "__main__":
    width = 32
    mlp = MLP([width, width])
    print(mlp)
    print(mlp.init_fn(jax.random.PRNGKey(0), 1))
