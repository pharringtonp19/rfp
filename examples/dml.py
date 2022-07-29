import os 
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
from functools import partial 
import numpy as np 

file_link = os.getcwd() + "/fig/"

flags.DEFINE_integer("init_key_num", 0, "initial key number")
flags.DEFINE_integer("n", 100, "number of observations")
flags.DEFINE_integer("features", 2, "number of features")
flags.DEFINE_float("lr", 0.01, "learning rate")
flags.DEFINE_integer("epochs", 1000, "epochs")
flags.DEFINE_bool("simulate", False, "simulate")
flags.DEFINE_integer("simulations", 5, "simulations")

FLAGS = flags.FLAGS

def outcome(d):
    return jnp.log(d**2 + 1.0 + jnp.sin(d * 1.5)) + 1.5

def main(argv):
    del argv

    @partial(jax.jit, static_argnums=(1))
    def simulate(init_key, plots: bool = False):
        # Data

        train_key, test_key, params_key = jax.random.split(init_key, 3)
        D, X, Y = batch_sample(partial(sample1, outcome), train_key, FLAGS.n, FLAGS.features)
        
        # Model 
        mlp = MLP([32, 1])
        params = mlp.init_fn(params_key, FLAGS.features)

        # Training 
        loss_fn = sqr_error(mlp).apply
        yuri = trainer(loss_fn, optax.sgd(learning_rate=FLAGS.lr, momentum=0.9), FLAGS.epochs)    
        opt_paramsD, lossesD = yuri.train(params, (X, D))
        opt_paramsY, lossesY = yuri.train(params, (X, Y))

        # Eval 
        D, X, Y = batch_sample(partial(sample1, outcome), test_key, FLAGS.n, FLAGS.features)
        residual_d = D - mlp.fwd_pass(opt_paramsD, X)
        residual_y = Y - mlp.fwd_pass(opt_paramsY, X)
        z = jnp.linalg.lstsq(residual_d, residual_y)[0]
        
        if plots:
            return z, lossesD, lossesY
        return z 

    if FLAGS.simulate:
        treatment = jnp.linspace(-3., 3, 1000).reshape(-1,1)
        regs = jnp.hstack((jnp.ones_like(treatment), treatment))
        y = jax.vmap(outcome)(treatment)
        target_coef = jnp.linalg.lstsq(regs, y)[0][1].item()
        print(target_coef)
        coeffs = jax.vmap(simulate)(jax.random.split(jax.random.PRNGKey(FLAGS.init_key_num), FLAGS.simulations)).squeeze()
        std = jnp.std(coeffs)
        array_plot = np.array((coeffs - target_coef)/ std)
        fig = plt.figure(dpi=300, tight_layout=True)
        plt.hist(array_plot, edgecolor='black', density=True, bins=20)
        plt.title('Density', loc='left', size=14)
        plt.xlabel(r'$(\hat{\theta} - \theta_0)/\sigma(\hat{\theta})$')
        filename = file_link + "dml.pdf"
        fig.savefig(filename, format="pdf")
        plt.show()
    
    else:
        coef, lossesD, lossesY = simulate(jax.random.PRNGKey(FLAGS.init_key_num), True)
        fig = plt.figure(dpi=300, tight_layout=True)
        plt.plot(lossesD)
        plt.plot(lossesY)
        plt.show()




if __name__ == "__main__":


    app.run(main)