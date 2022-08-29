import os

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pandas as pd
from absl import app, flags

from rfp import MLP, Trainer, ff1, time_grad

FLAGS = flags.FLAGS
flags.DEFINE_bool("person", False, "person")
flags.DEFINE_integer("init_key_num", 1, "Initial Key Number")
flags.DEFINE_integer("epochs", 100, "Epochs")
flags.DEFINE_float("lr", 0.1, "learning rate")
flags.DEFINE_bool("single_run", False, "Single Run of Training Loop")
flags.DEFINE_bool("multi_run", False, "Multi Run of Training Loop")
flags.DEFINE_bool("real_data", False, "Real Data")
flags.DEFINE_float("frac", 0.8, "Subsampling Fraction")
flags.DEFINE_integer("sims", 50, "Simulations")

np_file_link: str = os.getcwd() + "/examples/data/"


def main(argv):
    del argv

    ### TOY EXAMPLE
    if FLAGS.real_data:

        # READ IN THE DATASET
        if FLAGS.person:
            df = pd.read_csv(
                "/Users/patrickpower/Documents/GitHub/rfp/examples/data/evictions/direct_effect_person.csv"
            )
        else:
            df = pd.read_csv(
                "/Users/patrickpower/Documents/GitHub/rfp/examples/data/evictions/direct_effect_units.csv"
            )
        # DIVIDE DATASET INTO TREATED AND CONTROL GROUPS
        c_ts = df[df.Treated == 0]["eviction_rate"].values.reshape(-1, 1)
        c_ys = df[df.Treated == 0]["diff_outcome"].values.reshape(-1, 1)
        c_ws = df[df.Treated == 0]["fraction"].values.reshape(-1, 1)
        t_ts = df[df.Treated == 1]["eviction_rate"].values.reshape(-1, 1)
        t_ys = df[df.Treated == 1]["diff_outcome"].values.reshape(-1, 1)
        t_ws = df[df.Treated == 1]["fraction"].values.reshape(-1, 1)
        m1, m2 = jnp.max(c_ts), jnp.max(t_ts)
        t1 = jnp.maximum(m1, m2) + 0.01
        t2 = jnp.minimum(m1, m2) + 0.01  # For prediction (don't want to extrapolate)
        print(t1, t2)
        print(c_ts.shape, t_ts.shape)

    mlp = MLP([32, 1])
    params = mlp.init_fn(jax.random.PRNGKey(FLAGS.init_key_num), 1)
    loss_fn = ff1.Loss_fn(mlp)
    time_grad(loss_fn, params, (c_ys, c_ws, c_ts))
    yuri = Trainer(loss_fn, optax.adam(learning_rate=FLAGS.lr), FLAGS.epochs)

    def run(key, control_data, treated_data):

        ### Train
        params = mlp.init_fn(key, 1)
        c_opt_params, c_loss_history = yuri.train(params, (c_ys, c_ws, c_ts))
        t_opt_params, t_loss_history = yuri.train(params, (t_ys, t_ws, t_ts))

        ts = jnp.linspace(0, t2, 1000).reshape(-1, 1)  # prediction
        c_yhat = mlp.fwd_pass(c_opt_params, ts)
        t_yhat = mlp.fwd_pass(t_opt_params, ts)
        est_effect = t_yhat - c_yhat

        return c_loss_history, t_loss_history, ts, est_effect

    init_key = jax.random.PRNGKey(FLAGS.init_key_num)
    control_data = (c_ys, jnp.ones_like(c_ws), c_ts)
    treated_data = (t_ys, jnp.ones_like(t_ws), t_ts)

    if FLAGS.single_run:
        c_loss_history, t_loss_history, eviction_rt, est_effect = run(
            init_key, control_data, treated_data
        )
        np.save(
            np_file_link + f"ode1_loss_control_{FLAGS.person}_ff.npy",
            np.asarray(c_loss_history),
        )
        np.save(
            np_file_link + f"ode1_loss_treated_{FLAGS.person}_ff.npy",
            np.asarray(t_loss_history),
        )
        np.save(
            np_file_link + f"ode1_eviction_rt_{FLAGS.person}_ff.npy",
            np.asarray(eviction_rt),
        )
        np.save(
            np_file_link + f"ode1_est_effect_{FLAGS.person}_ff.npy",
            np.asarray(est_effect),
        )

    def simulate(init_key, frac, control_data, treated_data):
        subkey1, subkey2, subkey3 = jax.random.split(init_key, 3)
        c_ys, c_ws, c_ts = control_data
        t_ys, t_ws, t_ts = treated_data

        c_obs = int(np.floor(frac * len(c_ys)))
        t_obs = int(np.floor(frac * len(t_ys)))

        c_ys = jax.random.permutation(subkey1, jnp.arange(len(c_ys)))[:c_obs]
        c_ws = jax.random.permutation(subkey1, jnp.arange(len(c_ws)))[:c_obs]
        c_ts = jax.random.permutation(subkey1, jnp.arange(len(c_ts)))[:c_obs]

        t_ys = jax.random.permutation(subkey1, jnp.arange(len(t_ys)))[:t_obs]
        t_ws = jax.random.permutation(subkey1, jnp.arange(len(t_ws)))[:t_obs]
        t_ts = jax.random.permutation(subkey1, jnp.arange(len(t_ts)))[:t_obs]

        control_data = (c_ys, c_ws, c_ts)
        treated_data = (t_ys, t_ws, t_ts)
        _, _, _, est_effect = run(subkey3, control_data, treated_data)
        return est_effect

    if FLAGS.multi_run:
        es = jax.vmap(
            lambda key: simulate(key, FLAGS.frac, control_data, treated_data)
        )(jax.random.split(init_key, FLAGS.sims))
        np.save(
            np_file_link + f"ode1_multi_run_{FLAGS.frac}_{FLAGS.person}_ff.npy",
            np.asarray(es),
        )


if __name__ == "__main__":
    app.run(main)
