import os 
import jax 
import jax.numpy as jnp 
from absl import app, flags
from typing import Any, Type
from diffrax import Heun 
import optax 
import numpy as np 
from rfp import sample3, MLP, neuralODE, supervised_loss_time, linear_model_time, trainer

np_file_link : str =  os.getcwd() + "/examples/data/"

flags.DEFINE_integer("init_key_num", 0, "initial key number")
flags.DEFINE_integer("n", 200, "number of observations")
flags.DEFINE_integer("features", 2, "number of features")
flags.DEFINE_float("lr", 0.01, "learning rate")
flags.DEFINE_float("t1", 1., "length of integration interval")
flags.DEFINE_float("reg_val", 0., "Strength of Regularization")
flags.DEFINE_integer("epochs", 1000, "epochs")
flags.DEFINE_bool("single_run", False, "single run")
flags.DEFINE_bool("multi_run", False, "multi run")
flags.DEFINE_integer("simulations", 3000, "simulations")

FLAGS: Any = flags.FLAGS


def main(argv) -> None:
    del argv

    train_key, test_key, params_key = jax.random.split(jax.random.PRNGKey(FLAGS.init_key_num), 3)
    Y, D, T, X = jax.vmap(sample3, in_axes=(0, None))(jax.random.split(train_key, FLAGS.n), FLAGS.features)
    Y, D, T = Y.reshape(-1,1), D.reshape(-1,1), T.reshape(-1,1)
    print(Y.shape, D.shape, T.shape, X.shape)

    mlp = MLP([32, FLAGS.features])
    solver = Heun() 
    t1 = FLAGS.t1 

    params = mlp.init_fn(params_key, FLAGS.features + 1)
    feature_map = neuralODE(mlp, solver, t1)
    loss_fn = supervised_loss_time(linear_model_time, feature_map, FLAGS.reg_val, True)
    z = loss_fn(params, (Y, D, T, X))
    yuri = trainer(loss_fn, optax.sgd(learning_rate=FLAGS.lr, momentum=0.9), FLAGS.epochs)
    opt_params, (_, (prediction_loss, regularization)) = yuri.train(params, (Y, D, T, X))
    np.save(np_file_link+f"method_svl_prediction.npy", np.asarray(prediction_loss))
    np.save(np_file_link+f"method_svl_reg.npy", np.asarray(regularization))

    def get_coeff(feature_map, params, data):
        Y, D, T, X = data 
        phiX, _ = feature_map(params, X)
        regressors = jnp.hstack((D*T, D, T, jnp.ones_like(D), phiX))
        coeff = jnp.linalg.lstsq(regressors, Y)[0][0]
        return coeff

    z = get_coeff(feature_map, opt_params, (Y, D, T, X))
    print(z)
if __name__ == "__main__":
    app.run(main)