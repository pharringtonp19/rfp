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
from rfp._src.types import Params, Data


@dataclass(frozen=True, slots=True)
class trainer:
    loss_fn: callable # This is actually not a callable!
    opt: optax.GradientTransformation
    epochs: int 

    def train(self, params: Params, data: Data):
        """Params and Data"""
        def update_fn(carry, t):
            params, opt_state = carry
            loss_values, grads = jax.value_and_grad(self.loss_fn, has_aux=self.loss_fn.aux_status)(params, data)
            updates, opt_state = self.opt.update(grads, opt_state, params)
            params = optax.apply_updates(params, updates)
            return (params, opt_state), loss_values

        (opt_params, _), loss_values_history = jax.lax.scan(
            update_fn, (params, self.opt.init(params)), xs=None, length=self.epochs
        )
        return opt_params, loss_values_history

