"""
In my scripts, I label an instance of the trainer
class as "Yuri" in a reference to the great film 
The Pink Panther. If you recall from the movie,
Yuri is the tainer who trains! 
"""

import jax
import optax

class trainer:


    def __init__(self, loss_fn: callable, opt, epochs: int):
        """Training """
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

