from dataclasses import dataclass

import jax
import jax.numpy as jnp
from diffrax import Heun, ODETerm, PIDController, SaveAt, diffeqsolve

from rfp._src.nn import MLP
from rfp._src.utils import Model_Params


def init_params(key, mlp):
    subkey1, subkey2 = jax.random.split(key, 2)
    other = jax.random.normal(subkey1, shape=(1,))
    body = mlp.init_fn(subkey2, 2)
    return Model_Params(body, other)


def dynamics(mlp, t, y, args):
    state = jnp.hstack((y, t))
    return mlp.fwd_pass(args, state)


def Path(dynamics, params):
    return diffeqsolve(
        ODETerm(dynamics),
        Heun(),
        t0=0,
        t1=1,
        dt0=None,
        y0=params.other,
        stepsize_controller=PIDController(rtol=1e-3, atol=1e-3),
        args=params.body,
        saveat=SaveAt(dense=True),
    )


def predict(path, x):
    return path.evaluate(x)


@dataclass
class Loss_fn:
    dynamics: callable
    aux_status: bool = False

    def __call__(self, params, data):
        ys, ws, xs = data
        path = Path(self.dynamics, params)
        yhat = jax.vmap(predict, in_axes=(None, 0))(path, xs).reshape(
            -1,
        )
        return jnp.mean(ws * (yhat - ys) ** 2)


# def relative_dynamics(path, t, y, args):
#   m = lambda t: jnp.reshape(predict(path, t), ())
#   return dynamics(t, y, args) * jax.grad(m)(t) # jac seems unnecessary!
