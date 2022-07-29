import jax
import optax


def init_update_fn(loss_fn, opt, data):
    @jax.jit
    def update_fn(carry):
        params, opt_state = carry
        loss, grads = jax.value_and_grad(loss_fn)(params, data)
        updates, opt_state = opt.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return (params, opt_state), loss

    return update_fn


def init_train_fn(loss_fn, opt, epochs, data):
    @jax.jit
    def train_fn(params):
        def update_fn(carry, t):
            params, opt_state = carry
            loss, grads = jax.value_and_grad(loss_fn)(params, data)
            updates, opt_state = opt.update(grads, opt_state, params)
            params = optax.apply_updates(params, updates)
            return (params, opt_state), loss

        (opt_params, _), losses = jax.lax.scan(
            update_fn, (params, opt.init(params)), xs=None, length=epochs
        )
        return opt_params, losses

    return train_fn

