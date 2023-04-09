import jax 
import jax.numpy as jnp
import optax 
from optax import set_to_zero 
from rfp import MLP, Model_Params, Cluster_Loss, Supervised_Loss, loss_fn_real, Trainer, predict 
import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams['image.interpolation'] = 'nearest'
rcParams['image.cmap'] = 'viridis'
rcParams['axes.grid'] = False
from absl import app, flags
import os 
from pathlib import Path

github_folder = str(Path(os.getcwd()).parent.absolute())
file_link: str = github_folder + "/rfp_paper/figures/simulations/"
print(github_folder)
print(file_link)

flags.DEFINE_integer("init_key_num", 0, "Initial key number")
flags.DEFINE_integer("n", 30, "Number of observations per cluster")
flags.DEFINE_integer("c", 2, "Number of clusters")
flags.DEFINE_integer("f", 1, "Number of Features")
flags.DEFINE_integer("net_width", 32, "Width of Neural Network")
flags.DEFINE_float("reg_val", 0.0, "Regularization Value")
flags.DEFINE_bool("full", False, "Full Body Training")
FLAGS = flags.FLAGS


def plot_data(batch, predict=None):
  X, Y = batch
  fig = plt.figure(dpi=300, tight_layout=True, figsize=(6.4, 4.8))
  if predict is not None:
    plt.plot(predict[0], predict[1], color='black', linewidth=3)
  for i in range(X.shape[0]):
    plt.scatter(X[i], Y[i], label=i)
  plt.legend(title=r'$E[Y|C=i]$', ncol=2, fontsize='xx-small', frameon=False)
  plt.title(r'$Y$', loc='left')
  plt.xlabel(r'$X$')
  fig.savefig(file_link + f'motivating_cefs_{len(X)}_{predict[2]}_{predict[3]}.png', format='png')
  plt.show()

def polynomial(weights, x):
    return weights[0] + weights[1]*x + .5*weights[2]*x**2 + (1/6)*weights[3]*x**3

def get_data(key, n_obs):
  key, subkey1, subkey2, subkey3 = jax.random.split(key, 4)
  x_start = jax.random.uniform(subkey1, shape=(1,), minval=-3., maxval=-2.)
  lenth = jax.random.uniform(subkey2, shape=(1,), minval=1., maxval=4.)
  x = jnp.linspace(x_start, x_start + lenth, n_obs)
  weights = jax.random.normal(subkey3, shape=(4,))
  y = jax.vmap(polynomial, in_axes=(None, 0))(weights, x)
  return x, y


def main(argv) -> None:
    del argv

    batch = jax.vmap(get_data, in_axes=(0, None))(jax.random.split(jax.random.PRNGKey(2), FLAGS.c), FLAGS.n)
    batch = ((batch[0] - jnp.mean(batch[0]))/jnp.std(batch[0]), batch[1])

    width = FLAGS.net_width
    mlp = MLP([width, width])

    supervised_loss = Supervised_Loss(loss_fn = loss_fn_real, 
                                        feature_map = mlp.embellished_fwd_pass,
                                        reg_value = 0.0, 
                                        aux_status = False)


    key1, key2 = jax.random.split(jax.random.PRNGKey(FLAGS.init_key_num))
    model_params = Model_Params(mlp.init_fn(key1, 1),                      # Base
                                jax.random.normal(key2, shape=(width,)),   # Head
                                jnp.zeros(shape=()))                                        # Bias

    Y = jnp.vstack([jnp.expand_dims(batch[1][i],0) for i in range(FLAGS.c)])
    X = jnp.vstack([jnp.expand_dims(batch[0][i],0) for i in range(FLAGS.c)])
    W = jnp.ones_like(X)
    cluster_data = {'Y': Y, 'X':X, 'Weight': W}                          


    if FLAGS.full:
      inner_opt = optax.sgd(learning_rate=0.001)
    else:
      inner_opt = optax.multi_transform({"train": optax.sgd(learning_rate=0.001), "zero": set_to_zero()},
                            Model_Params("zero", "train", "train"))

    inner_yuri = Trainer(loss_fn = supervised_loss, 
                        opt = inner_opt, 
                        epochs = 3)

    cluster_loss = Cluster_Loss(inner_yuri, reg_value=FLAGS.reg_val, aux_status=False)

    yuri = Trainer(cluster_loss, optax.sgd(learning_rate=0.001, momentum=0.9), 1000)

    print(cluster_loss(model_params, cluster_data))
    opt_params, _ = yuri.train(model_params, cluster_data)
    print(cluster_loss(opt_params, cluster_data))
    xs = jnp.linspace(jnp.min(batch[0]), jnp.max(batch[0]), 1000).reshape(-1,1)
    yhat_post = predict(False, mlp.fwd_pass, opt_params, xs)

    plot_data(batch, (xs, yhat_post, FLAGS.reg_val, FLAGS.full))


if __name__ == "__main__":
    app.run(main)
