import os
from functools import partial

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax
import pandas as pd
from absl import app, flags

from rfp import (
    MLP,
    Cluster_Loss_ff,
    Trainer,
    batch_sample_weight,
    ff1,
    sample4,
    split_weight,
    time_grad,
    parallel
)

FLAGS = flags.FLAGS
flags.DEFINE_bool("person", False, "person")
flags.DEFINE_integer("init_key_num", 1, "Initial Key Number")
flags.DEFINE_integer("epochs", 100, "Epochs")
flags.DEFINE_integer("inner_epochs", 3, "Epochs")
flags.DEFINE_float("lr", 0.001, "learning rate")
flags.DEFINE_float("inner_lr", 0.001, "learning rate")
flags.DEFINE_bool("single_run", False, "Single Run of Training Loop")
flags.DEFINE_bool("multi_run", False, "Multi Run of Training Loop")
flags.DEFINE_bool("real_data", False, "Real Data")
flags.DEFINE_float("frac", 0.8, "Subsampling Fraction")
flags.DEFINE_float("reg_value", 0.9, "Subsampling Fraction")
flags.DEFINE_integer("sims", 40, "Simulations")
flags.DEFINE_integer("clusters", 10, "clusters")
flags.DEFINE_integer("n", 100, "obs")

np_file_link: str = os.getcwd() + "/examples/data/"


def main(argv):
    del argv

    # DATA
    data_key, params_key = jax.random.split(jax.random.PRNGKey(FLAGS.init_key_num))
    data = {}
    for i in range(FLAGS.clusters):
        ys, ws, ts, ds = batch_sample_weight(FLAGS.n)(sample4)(data_key)
        cluster_data = jnp.hstack((ys, ws, ts))
        data_key, _ = jax.random.split(data_key)
        data[i] = cluster_data

    # plt.scatter(data[0][:,2], data[0][:,0])
    # plt.scatter(data[1][:,2], data[1][:,0])
    # plt.show()

    mlp = MLP([32, 1])
    params = mlp.init_fn(jax.random.PRNGKey(FLAGS.init_key_num), 1)
    loss_fn = ff1.Loss_fn(mlp)

    print("timing single cluster")
    time_grad(loss_fn, params, data[0])
    inner_yuri = Trainer(
        loss_fn, optax.sgd(learning_rate=FLAGS.inner_lr), FLAGS.inner_epochs
    )
    cluster_loss = Cluster_Loss_ff(inner_yuri, FLAGS.reg_value, False)
    time_grad(cluster_loss, params, data)
    yuri = Trainer(cluster_loss, optax.sgd(learning_rate=FLAGS.lr), FLAGS.epochs)

    @parallel.pjit_key(FLAGS.sims)
    def run(key):

        ### Train
        params = mlp.init_fn(key, 1)
        opt_params, loss_history = yuri.train(params, data)
        # c_opt_params, c_loss_history = yuri.train(params, (c_ys, c_ws, c_ts))
        # t_opt_params, t_loss_history = yuri.train(params, (t_ys, t_ws, t_ts))

        ts = jnp.linspace(0, 1, 1000).reshape(-1, 1)  # prediction
        yhat = mlp.fwd_pass(opt_params, ts)
        # c_yhat = mlp.fwd_pass(c_opt_params, ts)
        # t_yhat = mlp.fwd_pass(t_opt_params, ts)
        # est_effect = t_yhat - c_yhat

        return loss_history, ts, yhat  # c_loss_history, t_loss_history, ts, est_effect
    
    loss_history, ts, yhat = run(jax.random.PRNGKey(0))
    print(type(loss_history), len(loss_history))
    # plt.plot(loss_history)
    # plt.show()

    # plt.plot(ts, yhat)
    # for i in range(FLAGS.clusters):
    #     plt.scatter(data[i][:, 2], data[i][:, 0])
    # plt.show()

    # init_key = jax.random.PRNGKey(FLAGS.init_key_num)
    # control_data = (c_ys, jnp.ones_like(c_ws), c_ts)
    # treated_data = (t_ys, jnp.ones_like(t_ws), t_ts)

    # if FLAGS.single_run:
    #     c_loss_history, t_loss_history, eviction_rt, est_effect = run(
    #         init_key, control_data, treated_data
    #     )
    #     np.save(
    #         np_file_link + f"ode1_loss_control_{FLAGS.person}_ff.npy",
    #         np.asarray(c_loss_history),
    #     )
    #     np.save(
    #         np_file_link + f"ode1_loss_treated_{FLAGS.person}_ff.npy",
    #         np.asarray(t_loss_history),
    #     )
    #     np.save(
    #         np_file_link + f"ode1_eviction_rt_{FLAGS.person}_ff.npy",
    #         np.asarray(eviction_rt),
    #     )
    #     np.save(
    #         np_file_link + f"ode1_est_effect_{FLAGS.person}_ff.npy",
    #         np.asarray(est_effect),
    #     )

    # def simulate(init_key, frac, control_data, treated_data):
    #     subkey1, subkey2, subkey3 = jax.random.split(init_key, 3)
    #     c_ys, c_ws, c_ts = control_data
    #     t_ys, t_ws, t_ts = treated_data

    #     c_obs = int(np.floor(frac * len(c_ys)))
    #     t_obs = int(np.floor(frac * len(t_ys)))

    #     c_ys = jax.random.permutation(subkey1, jnp.arange(len(c_ys)))[:c_obs]
    #     c_ws = jax.random.permutation(subkey1, jnp.arange(len(c_ws)))[:c_obs]
    #     c_ts = jax.random.permutation(subkey1, jnp.arange(len(c_ts)))[:c_obs]

    #     t_ys = jax.random.permutation(subkey1, jnp.arange(len(t_ys)))[:t_obs]
    #     t_ws = jax.random.permutation(subkey1, jnp.arange(len(t_ws)))[:t_obs]
    #     t_ts = jax.random.permutation(subkey1, jnp.arange(len(t_ts)))[:t_obs]

    #     control_data = (c_ys, c_ws, c_ts)
    #     treated_data = (t_ys, t_ws, t_ts)
    #     _, _, _, est_effect = run(subkey3, control_data, treated_data)
    #     return est_effect

    # if FLAGS.multi_run:
    #     es = jax.vmap(
    #         lambda key: simulate(key, FLAGS.frac, control_data, treated_data)
    #     )(jax.random.split(init_key, FLAGS.sims))
    #     np.save(
    #         np_file_link + f"ode1_multi_run_{FLAGS.frac}_{FLAGS.person}_ff.npy",
    #         np.asarray(es),
    #     )


if __name__ == "__main__":
    app.run(main)
