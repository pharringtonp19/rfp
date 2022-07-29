"""

Examples:
    >>> import jax 
    >>> import jax.numpy as jnp 
    >>> import flax.linen as nn
    >>> from typing import Sequence
    >>> from flax.core import unfreeze

    >>> mlp = MLP([32, 1], activation=nn.relu)
    >>> n, d = 100, 10 # (observations, features) 
    >>> xs = jax.random.normal(jax.random.PRNGKey(0), (n, d))
    >>> params = mlp.init_fn(jax.random.PRNGKey(1), d)
    >>> yhat = mlp.fwd_pass(params, xs)
"""
import jax 
import jax.numpy as jnp 
import flax.linen as nn
from typing import Sequence
from flax.core import unfreeze

class MLP(nn.Module):
    features: Sequence[int]
    activation: callable = nn.softplus

    @nn.compact
    def __call__(self, x: jnp.ndarray):
        for feat in self.features[:-1]:
            x = self.activation(nn.Dense(feat)(x))
        x = nn.Dense(self.features[-1])(x)
        return x
    
    def fwd_pass(self, params, x):
        """The Forward Pass"""
        return self.apply({"params": params}, x)
    
    def init_fn(self, key, features):
        """Initialize Parameters"""
        params =  unfreeze(self.init(key, jnp.ones((1, features))))['params']
        return params
