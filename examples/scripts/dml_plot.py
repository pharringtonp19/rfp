import os
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from absl import app, flags

plt.rcParams["figure.figsize"] = (6, 4)


github_folder = str(Path(os.getcwd()).parent.absolute())
file_link: str = github_folder + "/eviction_paper/examples/"
np_file_link: str = os.getcwd() + "/examples/data/"

flags.DEFINE_bool("continuous", False, "continuous treatment")
FLAGS = flags.FLAGS


def main(argv) -> None:
    del argv

    results = np.load(np_file_link + f"dml_{FLAGS.continuous}.npy")
    div = jnp.max(results)
    fig = plt.figure(dpi=300, tight_layout=True)
    plt.hist(results, edgecolor="black", density=True, bins=40)
    plt.title("Density", loc="left", size=14)
    plt.xlabel(r"$(\hat{\theta} - \theta_0)/\sigma(\hat{\theta})$")
    plt.xlim(-div - 0.5, div + 0.5)
    plt.axvline(0, linestyle="--", color="black")
    plt.xlim(-6.0, 6.0)
    plt.ylim(0, 0.5)
    filename = file_link + f"dml__{FLAGS.continuous}.png"
    fig.savefig(filename, format="png")
    plt.show()


if __name__ == "__main__":
    app.run(main)
