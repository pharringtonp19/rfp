import os
from pathlib import Path
import jax
import jax.numpy as jnp
from absl import app, flags
from rfp import MLP, f1, sample1, sqr_error, trainer
import optax
import matplotlib.pyplot as plt
from functools import partial
import numpy as np
from typing import Any, Type


np_file_link: str = os.getcwd() + "/examples/data/"

flags.DEFINE_integer("init_key_num", 0, "initial key number")
flags.DEFINE_integer("n", 200, "number of observations")
flags.DEFINE_integer("features", 2, "number of features")
flags.DEFINE_float("lr", 0.01, "learning rate")
flags.DEFINE_integer("epochs", 1000, "epochs")
flags.DEFINE_bool("single_run", False, "single run")
flags.DEFINE_bool("multi_run", False, "multi run")
flags.DEFINE_integer("simulations", 3000, "simulations")

FLAGS: Any = flags.FLAGS


def main(argv) -> None:
    del argv

    @partial(jax.jit, static_argnums=(1))
    def simulate(init_key, plots: bool = False):

        # Data
        train_key, test_key, params_key = jax.random.split(init_key, 3)
        sampler = partial(sample1, f1)
        Y, D, X = jax.vmap(sampler, in_axes=(0, None))(
            jax.random.split(train_key, FLAGS.n), FLAGS.features
        )
        Y, D = Y.reshape(-1, 1), D.reshape(-1, 1)

        # Model
        mlp = MLP([32, 1])
        params = mlp.init_fn(params_key, FLAGS.features)

        # Training
        loss_fn = sqr_error(mlp)
        z = loss_fn(params, (Y, X))
        yuri = trainer(
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
        treatment = jnp.linspace(-3.0, 3, 1000).reshape(-1, 1)
        regs = jnp.hstack((jnp.ones_like(treatment), treatment))
        y = jax.vmap(f1)(treatment)
        target_coef = jnp.linalg.lstsq(regs, y)[0][1].item()
        print(target_coef)
        coeffs = jax.vmap(simulate)(
            jax.random.split(jax.random.PRNGKey(FLAGS.init_key_num), FLAGS.simulations)
        ).squeeze()
        std = jnp.std(coeffs)
        array_plot = np.array((coeffs - target_coef) / std)
        np.save(np_file_link + f"dml.npy", np.asarray(array_plot))

    if FLAGS.single_run:
        _, lossesD, lossesY = simulate(jax.random.PRNGKey(FLAGS.init_key_num), True)
        print(lossesD[-1], lossesY[-1])


if __name__ == "__main__":
    app.run(main)
