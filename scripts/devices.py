import jax
import jax.numpy as jnp
import chex


@jax.jit
@chex.assert_max_traces(n=1)
def fn_sum_jitted(x, y):
    return x + y


z = fn_sum_jitted(jnp.zeros(3), jnp.zeros(3))
t = fn_sum_jitted(jnp.zeros((6, 7)), jnp.zeros((6, 7)))  # AssertionError!
