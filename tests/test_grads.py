import jax 
import jax.numpy as jnp 
from jax.experimental import maps 
from jax.experimental import PartitionSpec 
from jax.experimental.pjit import pjit 
import numpy as np 
from rfp import MLP, time_grad, Sqr_Error
from functools import partial 
from einops import rearrange 
import timeit
from dataclasses import dataclass

from rfp._src import parallel
from rfp._src.utils import time_grad_pvmap

from absl import app, flags

flags.DEFINE_integer("n", 200, "number of observations")
flags.DEFINE_integer("features", 2, "number of features")
FLAGS = flags.FLAGS

def main(argv) -> None:
    del argv

    mlp = MLP([32, 32, 1])
    params = mlp.init_fn(jax.random.PRNGKey(0), FLAGS.features)
    input_data = np.arange(FLAGS.n * FLAGS.features).reshape(-1, FLAGS.features).astype(jnp.float32)
    targets = jnp.ones((FLAGS.n, 1))

    data = jnp.hstack((targets, input_data))

    def data_split(data):
      y = data[:,0].reshape(-1,1)
      x = data[:,1:]
      return y, x

    sqr_error = Sqr_Error(mlp, data_split)

    time_grad_pvmap(sqr_error, params, data)
    time_grad(sqr_error, params, data)

if __name__ == "__main__":
    app.run(main)