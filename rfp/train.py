"""
This is Yuri (from the Pinkpanther!) is the trainer who trains
"""

import jax
import optax

class trainer:

    def __init__(self, loss_fn, opt, epochs):
        self.loss_fn = loss_fn
        self.opt = opt 
        self.epochs = epochs 

    def train(self, params, data):
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

