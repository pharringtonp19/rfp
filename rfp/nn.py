from typing import Sequence, Callable
import jax
from flax import linen as nn
import jax.numpy as jnp
from flax.core import unfreeze
from dataclasses import dataclass

class MLP(nn.Module):
    nodes: Sequence[int]
    activation: Callable = nn.relu


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
    
@dataclass
class Model:
    fwd_pass_model: nn.Module
    final_activation: Callable = nn.sigmoid
    
    def fwd_pass(self, params, x):
        phiX = self.fwd_pass_model.fwd_pass(params.body, x)
        return self.final_activation(phiX @ params.head + params.bias)
    
    def embellished_fwd_pass(self, params, x):
        phiX, penalty = self.fwd_pass_model.embellished_fwd_pass(params.body, x)
        return self.final_activation( phiX @ params.head + params.bias), penalty
