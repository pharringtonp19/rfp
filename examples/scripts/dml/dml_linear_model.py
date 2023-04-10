import os
from functools import partial
from pathlib import Path
from typing import Any, Type

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax
from absl import app, flags

from rfp import MLP, VC2015, Trainer

np_file_link: str = os.getcwd() + "/examples/data/"

flags.DEFINE_integer("init_key_num", 0, "initial key number")
flags.DEFINE_integer("n", 200, "number of observations")
flags.DEFINE_integer("features", 2, "number of features")
flags.DEFINE_integer("simulations", 10000, "simulations")
flags.DEFINE_float("theta", 0.5, "constant treatment effect")

FLAGS: Any = flags.FLAGS


def main(argv) -> None:
    del argv

    # Print target coefficient
    target_coef = FLAGS.theta
    print(f"Target Coefficient: {target_coef:.2f}")

    @partial(jax.jit, static_argnums=(1))
    def simulate(init_key, plots: bool = False):

        # Generate simulated data
        train_key, test_key = jax.random.split(init_key, 2)
        Y, D, X = VC2015(train_key, FLAGS.theta, FLAGS.n, FLAGS.features)

        # Compute coefficient
        regs = jnp.hstack((D, jnp.ones_like(D), X))
        coeff = jnp.linalg.lstsq(regs, Y)[0][0]

        return coeff

    # Simulate data and save to file
    coeffs = jax.vmap(simulate)(
        jax.random.split(jax.random.PRNGKey(FLAGS.init_key_num), FLAGS.simulations)
    ).squeeze()
    std = jnp.std(coeffs)
    array_plot = np.array((coeffs - target_coef) / std)
    print(np.mean(array_plot), array_plot.shape)
    np.save(np_file_link + "dml_linear_comp.npy", np.asarray(array_plot))


if __name__ == "__main__":
    app.run(main)
