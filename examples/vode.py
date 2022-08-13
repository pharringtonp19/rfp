import jax 
import jax.numpy as jnp 
from rfp import f1, sample1 
from functools import partial 
import matplotlib.pyplot as plt
from typing import NamedTuple
from distrax import Normal 
import os 

from absl import app, flags


file_link: str = os.getcwd() + "/docs/fig/"

FLAGS = flags.FLAGS
flags.DEFINE_integer("n", 100, "Number of Observations")
flags.DEFINE_integer("init_key_num", 0, "Initial Key Number")
flags.DEFINE_integer("epochs", 1000, "Epochs")
flags.DEFINE_integer("inner_epochs", 10, "Inner Epochs")
flags.DEFINE_float("lr", 0.001, "Learning Rate")
flags.DEFINE_float("scale", 2, "Scale")
flags.DEFINE_float("reg", 1., "Reg")

def main(argv):
    del argv

    data_key, params_key, train_key = jax.random.split(jax.random.PRNGKey(FLAGS.init_key_num), 3)
    d, _, y = jax.vmap(partial(sample1, f1, features=1))(jax.random.split(data_key, FLAGS.n)) 
    d, y = d.reshape(-1,1), y.reshape(-1,1)
    from typing import NamedTuple

    class Params(NamedTuple):
        weight: jnp.ndarray
        bias: jnp.ndarray


    def init(rng) -> Params:
        """Returns the initial model params."""
        weights_key, bias_key = jax.random.split(rng)
        weight = jax.random.normal(weights_key, ())
        bias = jax.random.normal(bias_key, ())
        return Params(weight, bias)


    def loss_fn(params: Params, x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
        """Computes the least squares error of the model's predictions on x against y."""
        pred = params.weight * x + params.bias
        return jnp.mean((pred - y) ** 2)


    


    def step(params: Params, x: jnp.ndarray, y:jnp.ndarray, lr: jnp.ndarray) -> Params:

        def update(params: Params, lr: float) -> Params:
            """Performs one SGD update step on params using the given data."""
            loss, grad = jax.value_and_grad(loss_fn)(params, x, y)
            new_params = jax.tree_map(
                lambda param, g: param - g * lr, params, grad)
            return new_params, loss
        params, losses = jax.lax.scan(update, params, lr)
        return params, losses

 



    params = init(params_key)

    lrs = jnp.full(shape=(FLAGS.inner_epochs,), fill_value=FLAGS.lr)

    def my_predict(params, x, y, p, key):
        lrs = jax.random.bernoulli(key, p=p, shape=(FLAGS.inner_epochs,))*FLAGS.lr
        params_x, _ = step(params, x, y, lrs)
        return params_x.weight * x + params_x.bias 

    def batch_loss_fn(params, x, y, z, keys):
        yhats_inner = jax.vmap(my_predict, in_axes=(None, 0, 0, 0, 0))(params, x, y, z, keys)
        yhats_standard = jax.vmap(predict, in_axes=(None, 0))(params, x)
        return (1-FLAGS.reg)*jnp.mean((y-yhats_inner)**2) + FLAGS.reg*jnp.mean((y-yhats_standard)**2)
    
    def predict(params, x):
        return params.weight*x + params.bias 

 


    def train(params, data, key):
        y, d = data
        z = jax.nn.softmax(jax.vmap(Normal(0, 1).log_prob)(d), axis=0)*FLAGS.scale

        def update(params, key):
            loss, grads = jax.value_and_grad(batch_loss_fn)(params, d, y, z, jax.random.split(key, FLAGS.n))
            params = jax.tree_map(lambda param, g: param - g * FLAGS.lr, params, grads)
            return params, loss 
        opt_params, losses = jax.lax.scan(update, params, jax.random.split(key, FLAGS.epochs))
        return opt_params, losses

    opt_params, losses = train(params, (y, d), train_key)
    plt.plot(losses)
    plt.show()


    pred1 = opt_params.weight * d + opt_params.bias

    regs = jnp.hstack((jnp.ones_like(d), d))
    ols_coef = jnp.linalg.lstsq(regs, y)[0]
    params_ols = Params(ols_coef[1], ols_coef[0])
    pred2 = params_ols.weight * d + params_ols.bias

    d_true = jnp.linspace(-3., 3., 1000).reshape(-1,1)
    y_true = f1(d_true)
    regs = jnp.hstack((jnp.ones_like(d_true), d_true))
    theta = jnp.linalg.lstsq(regs, y_true)[0]
    params_theta = Params(theta[1], theta[0])
    pred3 = params_theta.weight * d + params_theta.bias 

    plt.plot(d, pred1, label='Method')
    plt.plot(d, pred2, label='OLS')
    plt.plot(d, pred3, label='Target')
    plt.scatter(d, jax.vmap(f1)(d))
    plt.legend(frameon=False)
    plt.show()

    z = jax.nn.softmax(jax.vmap(Normal(0, 1).log_prob)(d), axis=0)*FLAGS.scale
    yhat = jax.vmap(my_predict, in_axes=(None, 0, 0, 0, 0))(opt_params, d, y, z, jax.random.split(train_key, FLAGS.n))
    plt.scatter(d, y)
    plt.scatter(d, yhat)
    plt.show()

if __name__ == "__main__":
    app.run(main)


