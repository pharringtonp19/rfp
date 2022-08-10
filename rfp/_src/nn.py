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
import jax 
import jax.numpy as jnp 
import flax.linen as nn
from typing import Sequence
from flax.core import unfreeze
from rfp._src.base import Params, Array, Key  

class MLP(nn.Module):
    features: Sequence[int]
    activation: callable = nn.relu

    @nn.compact
    def __call__(self, x: Array) -> Array:
        for feat in self.features[:-1]:
            x = self.activation(nn.Dense(feat)(x))
        x = nn.Dense(self.features[-1])(x)
        return x
    
    def fwd_pass(self, params: Params, x: Array) -> Array:
        """The Forward Pass"""
        return self.apply({"params": params}, x)
    
    def init_fn(self, key: Key, features: int) -> Params:
        """Initialize Parameters"""
        params =  unfreeze(self.init(key, jnp.ones((1, features))))['params']
        return params