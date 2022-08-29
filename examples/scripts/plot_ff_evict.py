import os
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from absl import app, flags

github_folder = str(Path(os.getcwd()).parent.absolute())
print(github_folder)
file_link: str = github_folder + "/eviction_paper/results/adv/"
np_file_link: str = os.getcwd() + "/examples/data/"
print(file_link)
print(np_file_link)

FLAGS = flags.FLAGS
flags.DEFINE_bool("person", False, "person")


def main(argv):
    del argv

    loss_control = np.load(np_file_link + f"ode1_loss_control_{FLAGS.person}_ff.npy")
    loss_treated = np.load(np_file_link + f"ode1_loss_treated_{FLAGS.person}_ff.npy")
    fig = plt.figure(dpi=300, tight_layout=True)
    plt.plot(loss_control, label="Control")
    plt.plot(loss_treated, label="Treated")
    plt.title("Loss", size=14, loc="left")
    plt.xlabel("Epochs", size=14)
    plt.legend(frameon=False)
    filename = file_link + f"1ode_loss_{FLAGS.person}_ff.pdf"
    fig.savefig(filename, format="pdf")
    plt.show()

    xlabel = "Renter" if FLAGS.person else "Occupied Unit"
    est_effect = np.load(np_file_link + f"ode1_est_effect_{FLAGS.person}_ff.npy")
    annualized_effect = 52 * est_effect
    ts = np.load(np_file_link + f"ode1_eviction_rt_{FLAGS.person}_ff.npy")
    ts = ts
    print(np.max(ts))
    fig = plt.figure(dpi=300, tight_layout=True)
    plt.plot(ts, annualized_effect)
    plt.title("Evictions Per Year", size=14, loc="left")
    plt.xlabel(f"Eviction Rate per {xlabel}", size=14)
    plt.hlines(0, xmin=0, xmax=np.max(ts), color="black", linestyle="--")
    plt.legend(frameon=False)
    filename = file_link + f"1ode_est_effect_{FLAGS.person}_ff.pdf"
    fig.savefig(filename, format="pdf")
    plt.show()

    es = np.load(np_file_link + f"ode1_multi_run_0.8_{FLAGS.person}_ff.npy")
    annualized_es = es * 52
    mean_effect = jnp.mean(annualized_es, axis=0)
    std_effect = np.std(annualized_es, axis=0)
    upper = mean_effect + 2 * std_effect
    lower = mean_effect - 2 * std_effect

    xlabel = "Renter" if FLAGS.person else "Occupied Unit"
    fig = plt.figure(dpi=300, tight_layout=True, figsize=(8, 6))
    plt.plot(ts, mean_effect, color="tab:blue", label="Estimate")
    plt.plot(ts, upper, color="black", linestyle="--")
    plt.plot(ts, lower, color="black", linestyle="--")
    plt.title("Evictions per Year", size=14, loc="left")
    plt.xlabel(f"Eviction Rate per {xlabel}", size=14)
    plt.hlines(0, xmin=0, xmax=np.max(ts), color="purple", linestyle="-")
    plt.legend(frameon=False)
    filename = file_link + f"1ode_est_effect_sampling_{FLAGS.person}_ff.pdf"
    fig.savefig(filename, format="pdf")
    plt.show()


if __name__ == "__main__":
    app.run(main)
