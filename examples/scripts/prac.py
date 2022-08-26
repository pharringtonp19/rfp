from rfp import pjit_time_grad, MLP
import jax 
from functools import partial 

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

mlp = MLP([32, 1])

params = mlp.init_fn(jax.random.PRNGKey(0), 1)
x = jax.random.normal(jax.random.PRNGKey(1), shape=(100, 1))

f = partial(mlp.fwd_pass, params)

pjit_time_grad(f, x)