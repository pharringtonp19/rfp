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
from functools import partial 

from rfp import (
    MLP,
    Model_Params,
    Supervised_Loss_Time,
    Trainer,
    batch_sample_time,
    linear_model_time,
    linear_model_trainable_time,
    neuralODE,
    sample3,
    split,
    time_grad,
    pjit_time_grad
)

import warnings

from rfp._src.utils import pjit_time_grad
warnings.simplefilter(action='ignore', category=FutureWarning)


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
flags.DEFINE_bool("jnp_data", False, "dataset as a jnp.array")
flags.DEFINE_bool("grad_layer", False, "trainable final layer")

FLAGS: Any = flags.FLAGS


def main(argv) -> None:
    del argv

    train_key, test_key, params_key, batch_key = jax.random.split(
        jax.random.PRNGKey(FLAGS.init_key_num), 4
    )

    # DATA
    Y, D, T, X = batch_sample_time(FLAGS.n)(sample3)(train_key, FLAGS.features)
    if FLAGS.jnp_data:
        data = jnp.hstack((Y, D, T, X))
    else:
        data = (Y, D, T, X)

    mlp = MLP([32, FLAGS.features])
    solver = Heun()
    t1 = FLAGS.t1

    ode_key, linear_key = jax.random.split(params_key)
    ode_params = mlp.init_fn(ode_key, FLAGS.features + 1)
    linear_params = (
        jax.random.normal(linear_key, shape=(3 + FLAGS.features,))
        if FLAGS.grad_layer
        else None
    )
    params = Model_Params(ode_params, linear_params)
    feature_map = neuralODE(mlp, solver, t1)

    # LOSS FUNCTION
    linear_layer = (
        linear_model_trainable_time if FLAGS.grad_layer else linear_model_time
    )

    print(data.shape)

    # @jax.grad
    # def f(params, data):
    #     Y, D, T, X = split(data)
    #     regs = jnp.hstack((D, T, X))
    #     loss = (Y - jnp.dot(regs, params))**2
    #     return jnp.mean(loss)
    #f = lambda params, data : jnp.dot(data, params)


    # loss_fn = Supervised_Loss_Time(linear_layer, feature_map, FLAGS.reg_val, False)
    #me_params = jax.random.normal(jax.random.PRNGKey(0), shape=(4,))
    f = partial(feature_map, params['ode_params'])
    val = f(X)
    print(type(val))
    pjit_time_grad(f, X)
    # time_grad(loss_fn, params, data)

    # yuri = Trainer(
    #     loss_fn, optax.sgd(learning_rate=FLAGS.lr, momentum=0.9), FLAGS.epochs
    # )
    # opt_params, (_, (prediction_loss, regularization)) = jax.jit(yuri.train)(
    #     params, data
    # )
    # np.save(np_file_link + f"method_svl_prediction.npy", np.asarray(prediction_loss))
    # np.save(np_file_link + f"method_svl_reg.npy", np.asarray(regularization))

    # def get_coeff(feature_map, params, data):
    #     Y, D, T, X = split(data)
    #     phiX, _ = feature_map(params, X)
    #     regressors = jnp.hstack((D * T, D, T, jnp.ones_like(D), phiX))
    #     coeff = jnp.linalg.lstsq(regressors, Y)[0][0]
    #     return coeff

    # z = get_coeff(feature_map, opt_params["ode_params"], (Y, D, T, X))
    # print(z)


if __name__ == "__main__":
    app.run(main)
