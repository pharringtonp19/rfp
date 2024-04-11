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


@dataclass
class Trainer:
    loss_fn: Callable
    opt: optax.GradientTransformation
    epochs: int
    eval: bool = False

    # Train Function
    def train(self, params, X, Y, mask):
        
        def update_fn(carry, t):
            params, opt_state = carry
            loss_values, grads = jax.value_and_grad(self.loss_fn, has_aux=self.loss_fn.aux_status)(params, X, Y, mask)
            updates, opt_state = self.opt.update(grads, opt_state, params)
            params = optax.apply_updates(params, updates)
            return (params, opt_state), loss_values
        (opt_params, _), loss_values_history = jax.lax.scan(update_fn, (params, self.opt.init(params)), xs=None, length=self.epochs)
        return opt_params, loss_values_history
    
    def train_with_val(self, params, X, Y, mask, train_idx, val_idx):       

        def update_fn(carry, t):
            params, opt_params, val_loss_opt, opt_state = carry
            train_loss, grads = jax.value_and_grad(self.loss_fn, has_aux=self.loss_fn.aux_status)(params, X[train_idx], Y[train_idx], mask[train_idx])
            val_loss = self.loss_fn(params, X[val_idx], Y[val_idx], mask[val_idx])
            updates, opt_state = self.opt.update(grads, opt_state, params)
            params = optax.apply_updates(params, updates)
            opt_params = jax.lax.cond(val_loss < val_loss_opt, lambda : params, lambda: opt_params)
            val_loss_opt = jnp.where(val_loss < val_loss_opt, val_loss, val_loss_opt)
            return (params, opt_params, val_loss_opt, opt_state), (train_loss, val_loss)
        (params, opt_params, _, _), (train_loss_history, val_loss_history) = jax.lax.scan(update_fn, (params, params, jnp.array(jnp.inf), self.opt.init(params)), xs=None, length=self.epochs)
        return params, opt_params, train_loss_history, val_loss_history
