"""Non-Cluster Diff-in-Diff Application"""

import os
from typing import Any, Type

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax
from absl import app, flags
from chex import assert_shape
from diffrax import Heun

from rfp import (
    MLP,
    Cluster_Loss,
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
)

np_file_link: str = os.getcwd() + "/examples/data/"

flags.DEFINE_integer("init_key_num", 0, "initial key number")
flags.DEFINE_integer("n", 200, "number of observations")
flags.DEFINE_integer("features", 2, "number of features")
flags.DEFINE_float("lr", 0.01, "learning rate")
flags.DEFINE_float("t1", 1.0, "length of integration interval")
flags.DEFINE_float("reg_val_vector", 0.0, "Strength of Regularization")
flags.DEFINE_float("reg_val_cluster", 0.0, "Strength of Regularization")
flags.DEFINE_integer("epochs", 1000, "epochs")
flags.DEFINE_bool("single_run", False, "single run")
flags.DEFINE_bool("multi_run", False, "multi run")
flags.DEFINE_integer("simulations", 3000, "simulations")
flags.DEFINE_integer("batch_size", 32, "batch size")
flags.DEFINE_bool("grad_layer", False, "trainable final layer")
flags.DEFINE_integer("clusters", 2, "number of clusters")
flags.DEFINE_float("inner_lr", 0.01, "inner learning rate")
flags.DEFINE_integer("inner_epochs", 2, "inner epochs")

FLAGS: Any = flags.FLAGS


def main(argv) -> None:
    del argv

    train_key, test_key, params_key, batch_key = jax.random.split(
        jax.random.PRNGKey(FLAGS.init_key_num), 4
    )

    # DATA
    z = {}
    for i in range(FLAGS.clusters):
        Y, D, T, X = batch_sample_time(FLAGS.n)(sample3)(train_key, FLAGS.features)
        data = jnp.hstack((Y, D, T, X))
        train_key, _ = jax.random.split(train_key)
        z[i] = data

    data = z

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

    supervised_loss = Supervised_Loss_Time(
        linear_layer, feature_map, FLAGS.reg_val_vector, False
    )
    print("Supervised Loss")
    time_grad(supervised_loss, params, data[0])
    inner_trainer = Trainer(
        supervised_loss,
        optax.sgd(learning_rate=FLAGS.inner_lr),
        epochs=FLAGS.inner_epochs,
    )
    cluster_loss = Cluster_Loss(
        supervised_loss, inner_trainer, FLAGS.reg_val_vector, False
    )
    print("Cluster Loss")
    time_grad(cluster_loss, params, data)

    # time_grad(loss_fn, params, data)


if __name__ == "__main__":
    app.run(main)
