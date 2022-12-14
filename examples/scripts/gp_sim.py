import jax

jax.config.update("jax_enable_x64", True)
from functools import partial

import jax.numpy as jnp
from tinygp import GaussianProcess, kernels

from rfp import gp_data

Xkernel = kernels.ExpSquared(scale=1.5)
Ykernel = kernels.ExpSquared(scale=1.5)

f = partial(gp_data, Xkernel, Ykernel, 5)
f_vec = jax.vmap(f)

init_key = jax.random.PRNGKey(0)
keys = jax.random.split(jax.random.PRNGKey(0), 100)
batch_y, batch_x = f_vec(keys)

print(batch_y.shape, batch_x.shape)
