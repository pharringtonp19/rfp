import os

import functools
from typing import Optional

import numpy as np
import timeit
import jax
import jax.numpy as jnp

from jax.experimental import mesh_utils
from jax.sharding import PositionalSharding

devices = mesh_utils.create_device_mesh((4,))
sharding = PositionalSharding(devices)
x = jax.random.normal(jax.random.PRNGKey(0), (8192, 8192))
y = jax.device_put(x, sharding.reshape(-1,1))
z = jnp.sin(y)

jax.debug.visualize_array_sharding(x) # Implicit print statement
print(" ")
jax.debug.visualize_array_sharding(y)
print(" ")
jax.debug.visualize_array_sharding(z)

# print(timeit.repeat(stmt="jnp.sin(x)",
#           number=5,
#           repeat=5, 
#           globals=globals()))

# print(timeit.repeat(stmt="jnp.sin(y)",
#           number=5,
#           repeat=5, 
#           globals=globals()))
