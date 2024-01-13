# Reference: https://stackoverflow.com/questions/100003/what-are-metaclasses-in-python?rq=1

"""
In my scripts, I label an instance of the trainer
class as "Yuri" in a reference to the great film 
The Pink Panther. If you recall from the movie,
Yuri is the trainer who trains! 
"""

from dataclasses import dataclass
from functools import partial
from typing import Any, Callable, Dict

import jax
import jax.numpy as jnp 
import optax
from jax.experimental import checkify
from rfp.utils import Model_Params


@dataclass
class Trainer:
    loss_fn: Callable
    opt: optax.GradientTransformation
    epochs: int

    # Train Function
    def train(self, params: Model_Params, X, Y, mask):
        def update_fn(carry, t):
            params, opt_state = carry
            loss_values, grads = jax.value_and_grad(self.loss_fn, has_aux=self.loss_fn.aux_status)(params, X, Y, mask)
            updates, opt_state = self.opt.update(grads, opt_state, params)
            params = optax.apply_updates(params, updates)
            return (params, opt_state), loss_values
        (opt_params, _), loss_values_history = jax.lax.scan(update_fn, (params, self.opt.init(params)), xs=None, length=self.epochs)
        return opt_params, loss_values_history
    
@dataclass
class Trainer_With_Val:
    loss_fn: Callable
    opt: optax.GradientTransformation
    epochs: int

    # Train Function
    def train(self, params: Model_Params, X, Y, mask, X_val, Y_val, mask_val):
        def update_fn(carry, t):
            params, val_opt_params, opt_val_loss, opt_state = carry
            loss_values, grads = jax.value_and_grad(self.loss_fn, has_aux=self.loss_fn.aux_status)(params, X, Y, mask)
            val_losses = jax.vmap(self.loss_fn, in_axes=(None, 0, 0, 0))(params, X_val, Y_val, mask_val)
            updates, opt_state = self.opt.update(grads, opt_state, params)
            params = optax.apply_updates(params, updates)
            val_opt_params = jnp.where(val_losses < opt_val_loss, params, val_opt_params)
            opt_val_loss = jnp.where(val_losses < opt_val_loss, val_losses, opt_val_loss)
            return (params, val_opt_params, opt_val_loss, opt_state), (loss_values, val_losses)
        (final_train_params, val_opt_params, _, _), loss_values_history = jax.lax.scan(update_fn, (params, self.opt.init(params)), xs=None, length=self.epochs)
        return final_train_params, val_opt_params, loss_values_history
