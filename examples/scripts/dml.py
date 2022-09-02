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

from rfp import MLP, Sqr_Error, Trainer, f1, sample1

np_file_link: str = os.getcwd() + "/examples/data/"

flags.DEFINE_integer("init_key_num", 0, "initial key number")
flags.DEFINE_integer("n", 200, "number of observations")
flags.DEFINE_integer("features", 2, "number of features")
flags.DEFINE_float("lr", 0.01, "learning rate")
flags.DEFINE_integer("epochs", 1000, "epochs")
flags.DEFINE_bool("single_run", False, "single run")
flags.DEFINE_bool("multi_run", False, "multi run")
flags.DEFINE_integer("simulations", 3000, "simulations")
flags.DEFINE_bool("continuous", False, "continuous treatment")

FLAGS: Any = flags.FLAGS


def main(argv) -> None:
    del argv

    if FLAGS.continuous:
        treatment = jnp.linspace(-3.0, 3, 1000).reshape(-1, 1)
        regs = jnp.hstack((jnp.ones_like(treatment), treatment))
        y = jax.vmap(f1)(treatment)
        target_coef = jnp.linalg.lstsq(regs, y)[0][1].item()
    else:
        target_coef = f1(1.0) - f1(0.0)
    print(f"Target Coefficient: {target_coef:.2f}")

    @partial(jax.jit, static_argnums=(1))
    def simulate(init_key, plots: bool = False):

        # Data
        train_key, test_key, params_key = jax.random.split(init_key, 3)
        sampler = partial(sample1, FLAGS.continuous, f1)
        Y, D, X = jax.vmap(sampler, in_axes=(0, None))(
            jax.random.split(train_key, FLAGS.n), FLAGS.features
        )
        Y, D = Y.reshape(-1, 1), D.reshape(-1, 1)

        # Model
        mlp = MLP([32, 1])
        params = mlp.init_fn(params_key, FLAGS.features)

        # Training
        loss_fn = Sqr_Error(mlp)
        z = loss_fn(params, (Y, X))
        yuri = Trainer(
            loss_fn, optax.sgd(learning_rate=FLAGS.lr, momentum=0.9), FLAGS.epochs
        )
        opt_paramsD, lossesD = yuri.train(params, (D, X))
        opt_paramsY, lossesY = yuri.train(params, (Y, X))

        # Eval
        Y, D, X = jax.vmap(sampler, in_axes=(0, None))(
            jax.random.split(test_key, FLAGS.n), FLAGS.features
        )
        Y, D = Y.reshape(-1, 1), D.reshape(-1, 1)
        residual_d = D - mlp.fwd_pass(opt_paramsD, X)
        residual_y = Y - mlp.fwd_pass(opt_paramsY, X)
        z = jnp.linalg.lstsq(residual_d, residual_y)[0]

        if plots:
            return z, lossesD, lossesY
        return z

    if FLAGS.multi_run:
        coeffs = jax.vmap(simulate)(
            jax.random.split(jax.random.PRNGKey(FLAGS.init_key_num), FLAGS.simulations)
        ).squeeze()
        std = jnp.std(coeffs)
        array_plot = np.array((coeffs - target_coef) / std)
        np.save(np_file_link + f"dml_{FLAGS.continuous}.npy", np.asarray(array_plot))

    if FLAGS.single_run:
        estimated_coeff, lossesD, lossesY = simulate(
            jax.random.PRNGKey(FLAGS.init_key_num), True
        )
        print(f"Estimated Coefficient: {estimated_coeff.item():.2f}")
        plt.plot(lossesD)
        plt.plot(lossesY)
        plt.show()
        # print(lossesD[-1], lossesY[-1])


if __name__ == "__main__":
    app.run(main)
