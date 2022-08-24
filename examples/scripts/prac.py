import jax
import jax.numpy as jnp
from jax import random


@jax.jit
def jax_fn(x):
    y = random.randint(random.PRNGKey(0), (1000, 1000), 0, 100)
    y2 = y @ y
    x2 = jnp.sum(y2) * x
    return x2


print(jax_fn(2.0))
