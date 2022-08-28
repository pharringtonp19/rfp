import jax
import jax.numpy as jnp
from absl import app, flags

from rfp import MLP, Model_Params, ode1

FLAGS = flags.FLAGS
flags.DEFINE_integer("init_key_num", 0, "Initial Key Number")


### TOY EXAMPLE
c_fn = lambda x: jnp.sin(x * 5)
t_fn = lambda x: jnp.cos(x * 5)
ts = jnp.linspace(0, 1, 50)
c_ys = jax.vmap(c_fn)(ts)
t_ys = jax.vmap(t_fn)(ts)


def main(argv):
    del argv

    mlp = MLP([2, 1])
    params = ode1.init_params(jax.random.PRNGKey(FLAGS.init_key_num), mlp)
    print(params)


if __name__ == "__main__":
    app.run(main)
