import os
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from absl import app, flags

plt.rcParams["figure.figsize"] = (6, 4)


github_folder = str(Path(os.getcwd()).parent.absolute())
file_link: str = github_folder + "/jmp_paper/figures/framework/"
np_file_link: str = os.getcwd() + "/examples/data/"

flags.DEFINE_bool("original", False, "original")
flags.DEFINE_bool("linear", False, "linear")
flags.DEFINE_bool("standard", False, "standard")
FLAGS = flags.FLAGS


def main(argv) -> None:
    del argv

    if FLAGS.linear:
        results = np.load(np_file_link + f"dml_linear_comp.npy")
    elif FLAGS.standard:
        results = np.load(np_file_link + "dml_standard_nn.npy")
    else:
        results = np.load(np_file_link + "dml_original.npy")
    div = jnp.max(results)
    fig = plt.figure(dpi=300, tight_layout=True)
    plt.hist(results, edgecolor="black", density=True, bins=20)
    plt.title("Density", loc="left", size=14)
    plt.xlabel(r"$(\hat{\theta} - \theta_0)/\sigma(\hat{\theta})$")
    plt.xlim(-div - 0.5, div + 0.5)
    plt.axvline(0, linestyle="--", color="black")
    plt.xlim(-6.0, 6.0)
    plt.ylim(0, 0.5)
    if FLAGS.linear:
        filename = file_link + "dml_linear_comp.png"
    elif FLAGS.standard:
        filename = file_link + "dml_standard_nn.png"
    else:
        filename = file_link + f"dml_original.png"
    fig.savefig(filename, format="png")
    plt.show()


if __name__ == "__main__":
    app.run(main)
