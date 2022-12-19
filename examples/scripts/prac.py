import jax.numpy as jnp 
from jax import lax 
from functools import partial 
import jax 
import numpy as np 
from jax.experimental.maps import xmap 

# z = jnp.arange(20).reshape(4,5).astype(jnp.float32)
# print(z)

x = np.ones((5,4,3))
print(np.einsum('ijk -> kji', x))