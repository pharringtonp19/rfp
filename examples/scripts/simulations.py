import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import optax
from matplotlib import rcParams

from rfp import (
    MLP,
    Cluster_Loss,
    Model_Params,
    Supervised_Loss,
    Trainer,
    gp_data,
    loss_fn_real,
    predict,
)

rcParams["image.interpolation"] = "nearest"
rcParams["image.cmap"] = "viridis"
rcParams["axes.grid"] = False
import os
from functools import partial
from pathlib import Path

from absl import app, flags
from tinygp import kernels

github_folder = str(Path(os.getcwd()).parent.absolute())
file_link: str = github_folder + "/rfp_paper/figures/simulations/"
print(github_folder)
print(file_link)

flags.DEFINE_integer("init_key_num", 0, "Initial key number")
flags.DEFINE_integer("n", 30, "Number of observations per cluster")
flags.DEFINE_integer("c", 2, "Number of clusters")
flags.DEFINE_integer("f", 1, "Number of Features")
flags.DEFINE_integer("net_width", 32, "Width of Neural Network")
flags.DEFINE_integer("sims", 32, "Number of Simulations")
flags.DEFINE_integer("test_c", 100, "Number of Test Clusters")
flags.DEFINE_float("reg_val", 0.0, "Regularization Value")
FLAGS = flags.FLAGS


def main(argv) -> None:
    del argv

    Xkernel = kernels.ExpSquared(scale=1.5)
    Ykernel = kernels.ExpSquared(scale=1.5)
    data_key = jax.random.PRNGKey(FLAGS.init_key_num)
    Y, X = jax.vmap(partial(gp_data, Xkernel, Ykernel, FLAGS.n))(
        jax.random.split(data_key, FLAGS.c)
    )
    W = jnp.ones_like(X)


#     width = FLAGS.net_width
#     mlp = MLP([width, width])

#     supervised_loss = Supervised_Loss(loss_fn = loss_fn_real,
#                                         feature_map = mlp.embellished_fwd_pass,
#                                         reg_value = 0.0,
#                                         aux_status = False)

#     inner_yuri = Trainer(loss_fn = supervised_loss,
#                         opt = optax.sgd(learning_rate=0.001),
#                         epochs = 3)

#     cluster_loss = Cluster_Loss(inner_yuri, reg_value=FLAGS.reg_val, aux_status=False)

#     yuri = Trainer(cluster_loss, optax.sgd(learning_rate=0.001, momentum=0.9), 1000)


#     def simulate(c, key):
#       key0, key1, key2, key3 = jax.random.split(key, 4)
#       model_params = Model_Params(mlp.init_fn(key0, 1),
#                                 jax.random.normal(key1, shape=(width,)),
#                                 jnp.zeros(shape=()))

#       # Sample Clusters
#       train_batch = jax.vmap(partial(sample_param_fn, param_fn), in_axes=(0, None))(jax.random.split(key2, FLAGS.c), FLAGS.n)
#       test_batch = jax.vmap(partial(sample_param_fn, param_fn), in_axes=(0, None))(jax.random.split(key2, FLAGS.test_c), FLAGS.n)

#       Y = jnp.vstack([jnp.expand_dims(train_batch[1][i],0) for i in range(c)])
#       X = jnp.vstack([jnp.expand_dims(train_batch[0][i],0) for i in range(c)]) #
#       W = jnp.ones_like(X)
#       cluster_data = {'Y': Y, 'X':X, 'Weight': W}
#       opt_params, _ = yuri.train(model_params, cluster_data)
#       fwd = partial(predict, False, mlp.fwd_pass, opt_params)
#       yhat = jax.vmap(fwd)(test_batch[0])
#       loss = jnp.mean((yhat-test_batch[1].squeeze())**2)
#       return loss

#     loss = jax.vmap(partial(simulate, 2))(jax.random.split(jax.random.PRNGKey(1), 32))
#     print(loss)
#     print(jnp.nanmean(loss))

#     # losses = jax.vmap(partial(simulate, 2))(jax.random.split(jax.random.PRNGKey(FLAGS.init_key_num), FLAGS.sims))
#     # print(jnp.nanmean(losses))
#     # results = []
#     # for i in [2, 5, 10]:
#     #     losses = jax.vmap(partial(simulate, i))(jax.random.split(jax.random.PRNGKey(FLAGS.init_key_num), FLAGS.sims))
#     #     print(i, jnp.nanmean(losses))
#     #     results.append(jnp.nanmean(losses))
#     # plt.plot([2, 5, 10], results)
#     # plt.show()

if __name__ == "__main__":
    app.run(main)
