from dataclasses import dataclass
from functools import partial

import jax
import jax.numpy as jnp

from rfp.nn import MLP
from rfp.utils import split_weight


@dataclass
class Loss_fn:
    mlp: MLP
    aux_status: bool = False

    def __call__(self, params, data):
        ys, ws, ts = split_weight(data)
        yhat = self.mlp.fwd_pass(params, ts)
        return jnp.mean(ws * (yhat - ys) ** 2)
