import jax 
import jax.numpy as jnp 


n = 200
d = 5000

a = jnp.ones((n, d), jnp.float32)

def f(x):
    return jnp.sum(jax.vmap(jnp.outer)(x, x), axis=0)
