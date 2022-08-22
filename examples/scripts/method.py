"""Non-Cluster Diff-in-Diff Application"""

import os
from typing import Any, Type

import jax
import jax.numpy as jnp
import numpy as np
import optax
from absl import app, flags
from chex import assert_shape
from diffrax import Heun

from rfp import (
    MLP,
    batch_sample_time,
    linear_model_time,
    neuralODE,
    sample3,
    supervised_loss_time,
    time_grad,
    trainer,
    training_sampler,
)

np_file_link: str = os.getcwd() + "/examples/data/"

flags.DEFINE_integer("init_key_num", 0, "initial key number")
flags.DEFINE_integer("n", 200, "number of observations")
flags.DEFINE_integer("features", 2, "number of features")
flags.DEFINE_float("lr", 0.01, "learning rate")
flags.DEFINE_float("t1", 1.0, "length of integration interval")
flags.DEFINE_float("reg_val", 0.0, "Strength of Regularization")
flags.DEFINE_integer("epochs", 1000, "epochs")
flags.DEFINE_bool("single_run", False, "single run")
flags.DEFINE_bool("multi_run", False, "multi run")
flags.DEFINE_integer("simulations", 3000, "simulations")
flags.DEFINE_integer("batch_size", 32, "batch size")

FLAGS: Any = flags.FLAGS


def main(argv) -> None:
    del argv

    train_key, test_key, params_key, batch_key = jax.random.split(
        jax.random.PRNGKey(FLAGS.init_key_num), 4
    )

    # DATA
    Y, D, T, X = batch_sample_time(FLAGS.n)(sample3)(train_key, FLAGS.features)

    mlp = MLP([32, FLAGS.features])
    solver = Heun()
    t1 = FLAGS.t1

    params = mlp.init_fn(params_key, FLAGS.features + 1)
    feature_map = neuralODE(mlp, solver, t1)

    # LOSS FUNCTION
    loss_fn = supervised_loss_time(linear_model_time, feature_map, FLAGS.reg_val, True)
    time_grad(loss_fn, params, (Y, D, T, X))

    yuri = trainer(
        loss_fn,
        optax.sgd(learning_rate=FLAGS.lr, momentum=0.9),
        FLAGS.epochs,
        batch_key,
        training_sampler,
        FLAGS.batch_size,
    )
    opt_params, (_, (prediction_loss, regularization)) = jax.jit(yuri.train)(
        params, (Y, D, T, X)
    )
    np.save(np_file_link + f"method_svl_prediction.npy", np.asarray(prediction_loss))
    np.save(np_file_link + f"method_svl_reg.npy", np.asarray(regularization))

    def get_coeff(feature_map, params, data):
        Y, D, T, X = data
        phiX, _ = feature_map(params, X)
        regressors = jnp.hstack((D * T, D, T, jnp.ones_like(D), phiX))
        coeff = jnp.linalg.lstsq(regressors, Y)[0][0]
        return coeff

    z = get_coeff(feature_map, opt_params, (Y, D, T, X))
    print(z)


if __name__ == "__main__":
    app.run(main)
