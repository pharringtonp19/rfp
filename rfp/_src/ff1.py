from dataclasses import dataclass

import jax.numpy as jnp

from rfp._src.nn import MLP


@dataclass
class Loss_fn:
    mlp: MLP
    aux_status: bool = False

    def __call__(self, params, data):
        ys, ws, xs = data
        yhat = self.mlp.fwd_pass(params, xs)
        return jnp.mean(ws * (yhat - ys) ** 2)
