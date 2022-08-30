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

# jaxlib = [ {platform = 'linux', markers = "platform_machine == 'x86_64'", url = "https://storage.googleapis.com/jax-releases/cuda11/jaxlib-0.3.15%2Bcuda11.cudnn82-cp310-none-manylinux2014_x86_64.whl"},
#           {platform = 'darwin', markers = "platform_machine == 'x86_64'", url = "https://storage.googleapis.com/jax-releases/mac/jaxlib-0.3.15-cp310-none-macosx_10_14_x86_64.whl"} ]


@dataclass(frozen=True, slots=True)
class Trainer:
    loss_fn: callable  # This is actually not a callable!
    opt: optax.GradientTransformation
    epochs: int

    def train(self, params: Params, data: Data):
        """Params and Data"""

        def update_fn(carry, t):
            params, opt_state = carry
            loss_values, grads = jax.value_and_grad(
                self.loss_fn, has_aux=self.loss_fn.aux_status
            )(params, data)
            updates, opt_state = self.opt.update(grads, opt_state, params)
            params = optax.apply_updates(params, updates)
            return (params, opt_state), loss_values

        (opt_params, _), loss_values_history = jax.lax.scan(
            update_fn, (params, self.opt.init(params)), xs=None, length=self.epochs
        )
        return opt_params, loss_values_history
