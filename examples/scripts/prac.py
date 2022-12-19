<<<<<<< HEAD
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
=======
import jax
import jax.numpy as jnp
import numpy as np


def f(x, y):
    return 2 * x * y


x, y = 3, 4
i32_scalar = jax.ShapeDtypeStruct((), jnp.dtype("int32"))
f_something = jax.jit(f).lower(i32_scalar, i32_scalar).compile()
print(type(f_something))
>>>>>>> a65b3e54146f0d2c9b87e3531350b005191d9cca
