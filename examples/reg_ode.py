import jax
import jax.numpy as jnp 
import os
import matplotlib.pyplot as plt
import optax
from absl import app, flags
from diffrax import Heun, SaveAt, NoAdjoint

from rfp.data import sample2
from rfp.featuremap import neuralODE
from rfp.losses import feature_map_loss
from rfp.nn import MLP
from rfp.train import trainer

file_link: str = os.getcwd() + "/docs/fig/"

FLAGS = flags.FLAGS
flags.DEFINE_integer("n", 10, "Number of Observations")
flags.DEFINE_integer("features", 1, "Number of Features")
flags.DEFINE_integer("init_key_num", 0, "Initial Key Number")
flags.DEFINE_bool("plot", False, "Plot")
flags.DEFINE_float("lr", 0.001, "Learning Rate")
flags.DEFINE_integer("epochs", 1000, "Epochs")
flags.DEFINE_float("reg_val", 0.0, "Regularization Value")
flags.DEFINE_float("t1", 1.0, "Integral Duration")

def main(argv):
    del argv

    data_key, params_key = jax.random.split(jax.random.PRNGKey(FLAGS.init_key_num))
    y, x = sample2(data_key, FLAGS.n)

    mlp = MLP([32, 1])
    solver = Heun() 
    t1 = FLAGS.t1 

    params = mlp.init_fn(params_key, FLAGS.features + 1)
    feature_map = neuralODE(mlp, solver, t1)
    yhat, regs = feature_map(params, x)

    loss_fn = feature_map_loss(feature_map, FLAGS.reg_val, True)
    z = loss_fn(params, (y, x))
    yuri = trainer(loss_fn, optax.sgd(learning_rate=FLAGS.lr, momentum=0.9), FLAGS.epochs)
    opt_params, (_, (prediction_loss, regularization)) = yuri.train(params, (y, x))
    yhat, _ = feature_map(opt_params, x)

    feature_map = neuralODE(mlp, 
                            solver, 
                            FLAGS.t1, 
                            saveat=SaveAt(dense=True),
                            adjoint=NoAdjoint())
    def eval_path(params, x):
        sol = feature_map.solve_ivp(params, x)
        ts = jnp.linspace(0, FLAGS.t1, 25)
        return jax.vmap(sol.evaluate)(ts)[:,0]

    zs = jnp.hstack((x, jnp.ones_like(x)))
    xs = jax.vmap(eval_path, in_axes=(None, 0))(opt_params, zs)


    fig, (ax0, ax1, ax2) = plt.subplots(nrows=1, ncols=3, dpi=300, tight_layout=True, figsize=(12,6))
    ax0.plot(prediction_loss, label='MSE')
    ax0.plot(regularization, label='Regularization')
    ax0.legend(frameon=False)
    ax0.set_title('Loss', loc='left', size=14)
    ax0.set_xlabel('Epochs', size=14)
    ax1.plot(x, yhat, label='Prediction')
    ax1.scatter(x, y, label='Data', color='black')
    ax1.legend(frameon=False)
    ax1.set_title('Outcome', size=14, loc='left')
    ax1.set_xlabel('Input', size=14)
    for i in xs:
        ax2.plot(i)
    ax2.set_title('Input', size=14, loc='left')
    ax2.set_xlabel('Depth/Time', size=14)
    filename = file_link + f"reg_ode_{FLAGS.reg_val}.png"
    fig.savefig(filename, format="png")
    plt.show()


if __name__ == "__main__":
    app.run(main)
