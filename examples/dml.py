import jax 
import jax.numpy as jnp 
from absl import app, flags
from rfp.nn import MLP 
from rfp.utils import batch_sample, init_keys
from rfp.data import sample1
from rfp.losses import sqr_error
from rfp.train import trainer
import optax 
import matplotlib.pyplot as plt

flags.DEFINE_integer("init_key_num", 0, "initial key number")
flags.DEFINE_integer("n", 100, "number of observations")
flags.DEFINE_integer("features", 10, "number of features")
flags.DEFINE_float("lr", 0.01, "learning rate")
flags.DEFINE_integer("epochs", 1000, "epochs")
FLAGS = flags.FLAGS

def main(argv):
    del argv

    # Data
    data_key, params_key = init_keys(FLAGS.init_key_num)
    D, X, Y = batch_sample(sample1, data_key, FLAGS.n, FLAGS.features)

    # Model 
    mlp = MLP([32, 1])
    params = mlp.init_fn(params_key, FLAGS.features)

    # Training 
    loss_fn = sqr_error(mlp).eval
    yuri = trainer(loss_fn, optax.sgd(learning_rate=FLAGS.lr, momentum=0.9), FLAGS.epochs)
    opt_params, losses = yuri.train(params, (X, D))
    plt.plot(losses)
    plt.show()



if __name__ == "__main__":


    app.run(main)