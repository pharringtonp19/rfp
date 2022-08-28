from functools import partial

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import optax
import pandas as pd
from absl import app, flags

from rfp import MLP, Model_Params, Trainer, ode1, time_grad

FLAGS = flags.FLAGS
flags.DEFINE_integer("init_key_num", 1, "Initial Key Number")
flags.DEFINE_integer("epochs", 100, "Epochs")
flags.DEFINE_float("lr", 0.1, "learning rate")
flags.DEFINE_bool("plot_loss", False, "plot loss training")
flags.DEFINE_bool("plot_prediction", False, "plot prediction")
flags.DEFINE_bool("plot_effect", False, "plot treatment effect")
flags.DEFINE_bool("real_data", False, "Real Data")


def main(argv):
    del argv

    ### TOY EXAMPLE
    if FLAGS.real_data:

        df = pd.read_csv(
            "/Users/patrickpower/Documents/GitHub/rfp/examples/data/evictions/direct_effect.csv"
        )
    else:
        c_fn = lambda x: jnp.sin(x * 5)
        t_fn = lambda x: jnp.cos(x * 5)
        ts = jnp.linspace(0, 1, 50)
        c_ys = jax.vmap(c_fn)(ts)
        t_ys = jax.vmap(t_fn)(ts)
        ws = jnp.ones_like(ts)

    # mlp = MLP([24, 24, 1])
    # params = ode1.init_params(jax.random.PRNGKey(FLAGS.init_key_num), mlp)
    # loss_fn = ode1.Loss_fn(partial(ode1.dynamics, mlp))
    # time_grad(loss_fn, params, (c_ys, ws, ts))
    # yuri = Trainer(loss_fn, optax.adam(learning_rate=FLAGS.lr), FLAGS.epochs)

    # c_opt_params, c_loss_history = yuri.train(params, (c_ys, ws, ts))
    # t_opt_params, t_loss_history = yuri.train(params, (t_ys, ws, ts))

    # fig = plt.figure(dpi=300, tight_layout=True)
    # plt.plot(c_loss_history, label="Control")
    # plt.plot(t_loss_history, label="Treated")
    # plt.title("Loss", size=14, loc="left")
    # plt.xlabel("Epochs", size=14)
    # plt.legend(frameon=False)
    # if FLAGS.plot_loss:
    #     plt.show()
    # else:
    #     plt.close(fig)

    # c_path = ode1.Path(partial(ode1.dynamics, mlp), c_opt_params)
    # t_path = ode1.Path(partial(ode1.dynamics, mlp), t_opt_params)
    # c_yhat = jax.vmap(ode1.predict, in_axes=(None, 0))(c_path, ts).reshape(
    #     -1,
    # )
    # t_yhat = jax.vmap(ode1.predict, in_axes=(None, 0))(t_path, ts).reshape(
    #     -1,
    # )

    # fig = plt.figure(dpi=300, tight_layout=True)
    # plt.scatter(ts, c_ys, label="Control Target", color="blue")
    # plt.plot(ts, c_yhat, label="Control Prediction", color="blue")
    # plt.scatter(ts, t_ys, label="Treatment Target", color="green")
    # plt.plot(ts, t_yhat, label="Treatment Prediction", color="green")
    # plt.title("Outcome", size=14, loc="left")
    # plt.xlabel("Input", size=14)
    # plt.legend(frameon=False)
    # if FLAGS.plot_prediction:
    #     plt.show()
    # else:
    #     plt.close(fig)

    # if FLAGS.plot_effect:
    #     fig = plt.figure(dpi=300, tight_layout=True)
    #     plt.scatter(ts, t_ys - c_ys, label="Target", color="blue")
    #     plt.plot(ts, t_yhat - c_yhat, label="Prediction", color="green")
    #     plt.title("Treatment Effect", size=14, loc="left")
    #     plt.xlabel("Input", size=14)
    #     plt.legend(frameon=False)
    #     plt.show()


if __name__ == "__main__":
    app.run(main)
