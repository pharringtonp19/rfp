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
flags.DEFINE_integer("clusters", 10, "number of clusters")

FLAGS: Any = flags.FLAGS


def main(argv) -> None:
    del argv

    train_key, test_key, params_key, batch_key = jax.random.split(
        jax.random.PRNGKey(FLAGS.init_key_num), 4
    )

    # DATA
    data = {}
    for i in range(FLAGS.clusters):
        train_key, obs_key, sample_key = jax.random.split(train_key, 3)
        n = jax.random.choice(obs_key, jnp.array([10, 12, 15]))
        data[i] = batch_sample_time(n)(sample3)(train_key, FLAGS.features)
    from jax.tree_util import tree_structure

    print(tree_structure(data))
    # print(type(data))
    # print(type(data[0]))
    # perm = jax.random.permutation(train_key, len(data))[:5]
    # batch_data = dict((k, data[k]) for k in perm)
    # print(type(batch_data))


if __name__ == "__main__":
    app.run(main)
