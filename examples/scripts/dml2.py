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

from rfp import MLP, Sqr_Error, Trainer, simulated_data

np_file_link: str = os.getcwd() + "/examples/data/"

flags.DEFINE_integer("init_key_num", 0, "initial key number")
flags.DEFINE_integer("n", 200, "number of observations")
flags.DEFINE_integer("features", 2, "number of features")
flags.DEFINE_float("lr", 0.01, "learning rate")
flags.DEFINE_integer("epochs", 1000, "epochs")
flags.DEFINE_bool("single_run", False, "single run")
flags.DEFINE_bool("multi_run", False, "multi run")
flags.DEFINE_integer("simulations", 400, "simulations")
flags.DEFINE_float("theta", 0.5, "constant treatment effect")

FLAGS: Any = flags.FLAGS


def main(argv) -> None:
    del argv

    target_coef = FLAGS.theta

    print(f"Target Coefficient: {target_coef:.2f}")

    @partial(jax.jit, static_argnums=(1))
    def simulate(init_key, plots: bool = False):

        # Data
        train_key, test_key, params_key = jax.random.split(init_key, 3)

        if FLAGS.vc:
            Y, D, X = simulated_data.VC2015(
                train_key, FLAGS.theta, FLAGS.n, FLAGS.features
            )
        else:
            sampler = partial(simulated_data.sample1, simulated_data.f1)
            Y, D, X = jax.vmap(sampler, in_axes=(0, None))(
                jax.random.split(train_key, FLAGS.n), FLAGS.features
            )
            Y, D = Y.reshape(-1, 1), D.reshape(-1, 1)

        # Model
        mlp = MLP([32, 1])
        params = mlp.init_fn(params_key, FLAGS.features + 1)

        # Training
        loss_fn = Sqr_Error(mlp)
        yuri = Trainer(
            loss_fn, optax.sgd(learning_rate=FLAGS.lr, momentum=0.9), FLAGS.epochs
        )
        opt_params, losses = yuri.train(params, (Y, jnp.hstack((D, X))))

        # Eval
        Y, D, X = simulated_data.VC2015(test_key, FLAGS.theta, FLAGS.n, FLAGS.features)
        D1 = jnp.ones_like(D)
        D0 = jnp.zeros_like(D)

        Y1 = mlp.fwd_pass(opt_params, jnp.hstack((D1, X)))
        Y0 = mlp.fwd_pass(opt_params, jnp.hstack((D0, X)))

        effect = jnp.mean(Y1 - Y0)

        if plots:
            return effect, losses
        return effect

    if FLAGS.multi_run:
        coeffs = jax.vmap(simulate)(
            jax.random.split(jax.random.PRNGKey(FLAGS.init_key_num), FLAGS.simulations)
        ).squeeze()
        std = jnp.std(coeffs)
        print(f"Mean Estimate: {jnp.mean(coeffs)}")
        print(len(coeffs))
        array_plot = np.array((coeffs - target_coef) / std)
        np.save(
            np_file_link + "dml_standard_nn.npy",
            np.asarray(array_plot),
        )

    if FLAGS.single_run:
        estimated_coeff, losses = simulate(jax.random.PRNGKey(FLAGS.init_key_num), True)
        print(f"Estimated Coefficient: {estimated_coeff.item():.2f}")
        plt.plot(losses)
        plt.show()


if __name__ == "__main__":
    app.run(main)
