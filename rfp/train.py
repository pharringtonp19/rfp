"""
In my scripts, I label an instance of the trainer
class as "Yuri" in a reference to the great film 
The Pink Panther. If you recall from the movie,
Yuri is the tainer who trains! 
"""

import jax
from numpy import float32
import optax
import chex
from dataclasses import dataclass 
from rfp.base import Params, Data
from typing import Tuple 


@dataclass(frozen=True, slots=True)
class trainer:
    loss_fn: callable 
    opt: optax.GradientTransformation
    epochs: int 

    def train(self, params: Params, data: Data) -> Tuple(Params, [float32]):
        """Params and Data"""
        def update_fn(carry, t):
            params, opt_state = carry
            loss, grads = jax.value_and_grad(self.loss_fn)(params, data)
            updates, opt_state = self.opt.update(grads, opt_state, params)
            params = optax.apply_updates(params, updates)
            return (params, opt_state), loss

        (opt_params, _), losses = jax.lax.scan(
            update_fn, (params, self.opt.init(params)), xs=None, length=self.epochs
        )
        return opt_params, losses

