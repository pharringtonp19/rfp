import matplotlib.pyplot as plt
from absl import app, flags

from rfp import MLP, Model_Params

flags.DEFINE_integer("init_key_num", 0, "initial key number")
flags.DEFINE_integer("net_width", 32, "network width")


FLAGS = flags.FLAGS


def main(argv) -> None:
    del argv


def plot_data(batch):
    X, Y = batch
    fig = plt.figure(dpi=300, tight_layout=True, figsize=(6.4, 4.8))
    for i in range(X.shape[0]):
        plt.scatter(X[i], Y[i], label=i)
    plt.legend(title=r"$E[Y|C=i]$", ncol=2, fontsize="xx-small")
    plt.title(r"$Y$", loc="left")
    plt.xlabel(r"$X$")
    fig.savefig("motivating_cefs.pdf", format="pdf")
    plt.show()

    mlp = MLP([FLAGS.net_width, FLAGS.net_width])


if __name__ == "__main__":
    app.run(main)
