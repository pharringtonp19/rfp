"""
In my scripts, I label an instance of the trainer
class as "Yuri" in a reference to the great film 
The Pink Panther. If you recall from the movie,
Yuri is the tainer who trains! 
"""

from dataclasses import dataclass

import chex
import jax
import optax
from numpy import float32

from rfp._src.types import Data, Params
from rfp._src.utils import training_sampler


@dataclass(frozen=True, slots=True)
class trainer:
    loss_fn: callable  # This is actually not a callable!
    opt: optax.GradientTransformation
    epochs: int
    init_key: int = jax.random.PRNGKey(0)
    sampler: callable = training_sampler
    batch_size: int = 32

    def train(self, params: Params, data: Data):
        """Params and Data"""

        def update_fn(carry, key):
            params, opt_state = carry
            sample = self.sampler(self.batch_size, data, key=key)
            loss_values, grads = jax.value_and_grad(
                self.loss_fn, has_aux=self.loss_fn.aux_status
            )(params, sample)
            updates, opt_state = self.opt.update(grads, opt_state, params)
            params = optax.apply_updates(params, updates)
            return (params, opt_state), loss_values

        (opt_params, _), loss_values_history = jax.lax.scan(
            update_fn,
            (params, self.opt.init(params)),
            xs=jax.random.split(self.init_key),
        )
        return opt_params, loss_values_history
