import os
from functools import partial
from pathlib import Path
from typing import Any, Type

import jax
import jax.numpy as jnp
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import optax
from absl import app, flags
from matplotlib import pyplot as plt
from matplotlib import rcParams
from matplotlib.offsetbox import AnnotationBbox, OffsetImage

rcParams["image.interpolation"] = "nearest"
rcParams["image.cmap"] = "viridis"
rcParams["axes.grid"] = False
plt.style.use("seaborn-dark-palette")

from matplotlib import font_manager

locations = str(Path(os.getcwd()).absolute()) + "/styles/Newsreader"
font_files = font_manager.findSystemFonts(fontpaths=locations)
print(locations)
print(font_files[0])
for f in font_files:
    font_manager.fontManager.addfont(f)
plt.rcParams["font.family"] = "Newsreader"


from rfp import MLP, VC2015, Sqr_Error, Trainer

github_folder = str(Path(os.getcwd()).parent.absolute())
np_file_link: str = github_folder + "/rfp/examples/data/"
paper_link = github_folder + "/rfp_paper/figures/simulations/high_dim_clusters/"

flags.DEFINE_integer("init_key_num", 0, "initial key number")
flags.DEFINE_integer("n", 200, "number of observations")
flags.DEFINE_integer("features", 15, "number of features")  # This is the only change
flags.DEFINE_float("lr", 0.001, "learning rate")
flags.DEFINE_integer("epochs", 1000, "epochs")
flags.DEFINE_bool("single_run", False, "single run")
flags.DEFINE_bool("multi_run", False, "multi run")
flags.DEFINE_integer("simulations", 400, "simulations")

FLAGS: Any = flags.FLAGS

# Load Non-Clustered Data from .npz file
loaded_data = np.load(np_file_link + "evict_arrays_False.npz")
Y = loaded_data["arr1"]
X = loaded_data["arr2"]
C = loaded_data["arr3"]

loaded_data = np.load(np_file_link + "evict_arrays_True.npz")
Y_c = loaded_data["arr1"]
X_c = loaded_data["arr2"]
C_c = loaded_data["arr3"]


def normalize(Z):
    # Compute column means and standard deviations
    means = jnp.mean(Z, axis=0)
    stds = jnp.std(Z, axis=0)

    # Normalize the columns
    Z_norm = (Z - means) / stds

    return Z_norm


Y, X = normalize(Y), normalize(X)
print(Y.shape, X.shape)
Y_c, X_c = normalize(Y_c), normalize(X_c)
print(Y_c.shape, X_c.shape)


def main(argv) -> None:
    del argv

    @partial(jax.jit, static_argnums=(1))
    def simulate(init_key, i: int = 0):

        # Keys
        train_key, test_key, params_key = jax.random.split(init_key, 3)

        # Remove i-th element from Y and X
        Y_r = jnp.delete(Y, i, axis=0)  # remove i-th element from Y
        X_r = jnp.delete(X, i, axis=0)  # remove i-th row from X
        Y_cr = jnp.delete(Y_c, i, axis=0)  # remove i-th element from Y
        X_cr = jnp.delete(X_c, i, axis=0)  # remove i-th row from X

        # Model
        mlp = MLP([64, 32, 1])
        params = mlp.init_fn(params_key, FLAGS.features)
        params_c = mlp.init_fn(params_key, FLAGS.features + 1)

        # Training Setup
        loss_fn = Sqr_Error(mlp)
        yuri = Trainer(
            loss_fn, optax.sgd(learning_rate=FLAGS.lr, momentum=0.9), FLAGS.epochs
        )

        # Train on full data
        opt_params, losses = yuri.train(params, (Y, X))
        opt_params_c, losses_c = yuri.train(params_c, (Y_c, X_c))

        # Train on reduced data
        opt_params_r, losses_r = yuri.train(params, (Y_r, X_r))
        opt_params_cr, losses_cr = yuri.train(params_c, (Y_cr, X_cr))

        Y_hat = mlp.fwd_pass(opt_params, X)
        Y_hat_c = mlp.fwd_pass(opt_params_c, X_c)
        Y_hat_r = mlp.fwd_pass(opt_params_r, X)
        Y_hat_cr = mlp.fwd_pass(opt_params_cr, X_c)

        absolute_difference = jnp.abs(Y_hat - Y_hat_r)
        absolute_difference_c = jnp.abs(Y_hat_c - Y_hat_cr)
        # cluster_effect = jnp.mean(absolute_difference, where=Z==True)
        # non_cluster_effect = jnp.mean(absolute_difference, where=Z==False)

        return (
            losses,
            losses_c,
            losses_r,
            losses_cr,
        )  # jnp.mean(absolute_difference), jnp.mean(absolute_difference_c)

    if FLAGS.multi_run:
        coeffs = jax.vmap(simulate)(
            jax.random.split(jax.random.PRNGKey(FLAGS.init_key_num), FLAGS.simulations)
        ).squeeze()
        std = jnp.std(coeffs)
        print(f"Mean Estimate: {jnp.mean(coeffs)}")
        print(len(coeffs))
        array_plot = np.array((coeffs - target_coef) / std)
        np.save(
            np_file_link + "dml_standard_nn.npy",
            np.asarray(array_plot),
        )

    if FLAGS.single_run:
        colors = ["#36454F", "#007C92", "#800080"]
        a, b, c, d = simulate(jax.random.PRNGKey(FLAGS.init_key_num), True)
        # Create a figure with two subplots
        fig, axs = plt.subplots(
            nrows=2, ncols=1, sharex=True, figsize=(8, 6), dpi=300, tight_layout=True
        )

        # Plot the first array in the first subplot
        axs[0].plot(a, label="Full Data", color=colors[0])
        # Plot the second array in the first subplot
        axs[0].plot(c, label="Dropping One Observation", color=colors[1])
        axs[0].legend(frameon=False, title="Sample", loc="upper right", fontsize=10)

        # Compute the difference between the two arrays
        diff = a - c

        # Plot the difference in the second subplot
        axs[1].plot(diff, color=colors[2])

        # Set the title and axis labels
        axs[0].set_title("Sensitivity Plots", size=14, loc="center", pad=30)
        axs[0].text(0.0, 1.02, s="Training Loss", transform=axs[0].transAxes)
        axs[1].text(0.0, 1.02, s="Difference", transform=axs[1].transAxes)
        axs[1].set_xlabel("Training Epochs")
        for key in "left", "right", "top":
            axs[0].spines[key].set_visible(False)
            axs[1].spines[key].set_visible(False)

        # Display the plot
        fig.savefig(paper_link + "sensitivity_plot.png")  # save the figure to file
        plt.show()


if __name__ == "__main__":
    app.run(main)
