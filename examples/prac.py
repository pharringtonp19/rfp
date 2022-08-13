import jax 
import jax.numpy as jnp 

def f(x):
    y, z = jnp.sin(x), jnp.cos(x)
    breakpoint()
    return y * z 

f(2.)