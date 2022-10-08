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
from matplotlib.ticker import MaxNLocator

from rfp import (
    MLP,
    Cluster_Loss,
    Model_Params,
    Supervised_Loss,
    Trainer,
    loss_fn_real,
    simulated_data,
)

np_file_link: str = os.getcwd() + "/examples/data/"

flags.DEFINE_integer("init_key_num", 0, "initial key number")
flags.DEFINE_integer("n", 100, "number of observations")
flags.DEFINE_integer("features", 2, "number of features")
flags.DEFINE_integer("clusters", 10, "number of clusters")
flags.DEFINE_float("learning_rate", 0.001, "learning rate")
flags.DEFINE_integer("max_inner_epochs", 20, "max number of epochs")

FLAGS: Any = flags.FLAGS


def main(argv) -> None:
    del argv

    print("hi")

    train_key, test_key, params_key, batch_key = jax.random.split(
        jax.random.PRNGKey(FLAGS.init_key_num), 4
    )

    # DATA
    z = {}
    for i in range(FLAGS.clusters):
        Y, D, T, X = simulated_data.batch_sample_time(FLAGS.n)(simulated_data.sample3)(
            train_key, FLAGS.features
        )
        data = jnp.hstack((Y, D, T, X))
        train_key, _ = jax.random.split(train_key)
        z[i] = data

    data = z
    for value in data.values():
        print(value.shape)

    mlp = MLP([32, 32])

    subkey1, subkey2 = jax.random.split(params_key)

    supervised_loss = Supervised_Loss(
        loss_fn=loss_fn_real,
        feature_map=mlp.embellished_fwd_pass,
        reg_value=0.0,
        aux_status=False,
    )

    params = Model_Params(
        mlp.init_fn(subkey1, FLAGS.features + 2),
        jax.random.normal(subkey2, shape=(32,)),
    )

    labels = ["supervised_learning", "regularized_maml", "maml"]
    for v, l in zip([0.0, 0.9, 1.0], labels):
        results = []
        for i in range(FLAGS.max_inner_epochs):
            inner_yuri = Trainer(
                loss_fn=supervised_loss,
                opt=optax.sgd(learning_rate=FLAGS.learning_rate),
                epochs=i,
            )

            cluster_loss = Cluster_Loss(
                inner_yuri=inner_yuri, reg_value=v, aux_status=False
            )
            loss = cluster_loss(params, data)
            results.append(loss)
            print(cluster_loss(params, data))

        np.save(
            np_file_link + f"inner_epochs_{l}.npy",
            np.asarray(results),
        )


if __name__ == "__main__":
    app.run(main)
