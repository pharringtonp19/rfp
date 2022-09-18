from dataclasses import dataclass
from typing import Any

import chex
import jax
import jax.numpy as jnp
from diffrax import (  # type: ignore
    BacksolveAdjoint,
    Heun,
    ODETerm,
    PIDController,
    SaveAt,
    diffeqsolve,
)
from flax.core import unfreeze

from rfp._src.nn import MLP  # Is this necessary?
from rfp._src.types import Array, Kleisi, ODE_Solver, Params


def predict(feature_map, params, X, real=True):
    if real:
        return feature_map(params.body, X)[0] @ params.other
    else:
        logits = feature_map(params.body, X)[0] @ params.other
        return jax.nn.sigmoid(logits)


@dataclass
class NeuralODE:
    """Neural ODE"""

    mlp: MLP
    solver: ODE_Solver
    t1: float
    saveat: Any = SaveAt(t1=True)
    adjoint: Any = BacksolveAdjoint()

    def vector_field(self, t: float, y: Array, args: Params) -> Array:
        """mlp parameterized vector field"""
        state = jnp.hstack([y, jnp.array(t)])
        return self.mlp.fwd_pass(args, state)

    def aug_vector_field(self, t: float, state: Array, args: Params) -> Array:
        """Augmented mlp parameterized vector field"""
        y, _ = state[:-1], state[-1]
        a = self.vector_field(t, y, args)
        b = jnp.reshape(
            jnp.linalg.norm(jax.jacobian(self.vector_field)(t, y, args)), (1,)
        )
        c = jnp.concatenate((a, b))
        return c

    def solve_ivp(self, params: Params, input: Array) -> Array:
        """fwd pass of ode"""
        return diffeqsolve(
            ODETerm(self.aug_vector_field),
            self.solver,
            t0=0,
            t1=self.t1,
            y0=input,
            dt0=None,
            adjoint=self.adjoint,
            stepsize_controller=PIDController(rtol=1e-3, atol=1e-3),
            saveat=self.saveat,
            args=params,
        )

    def __call__(self, params: Params, X: Array) -> Kleisi[Array]:
        """Batched FWD Pass"""
        sols = jax.vmap(self.solve_ivp, in_axes=(None, 0))(
            params, jnp.hstack((X, jnp.zeros((X.shape[0], 1))))
        ).ys.squeeze()
        phiX = sols[:, :-1]
        regs = sols[:, -1]
        return phiX, jnp.mean(regs)


if __name__ == "__main__":
    import matplotlib.pyplot as plt  # type: ignore

    from rfp._src.nn import MLP

    mlp = MLP([32, 1])
    params = mlp.init_fn(jax.random.PRNGKey(0), 2)
    x = jnp.linspace(-3, 3, 10).reshape(-1, 1)
    feature_map = neuralODE(mlp, Heun(), 1.0)
    yhat, regs = feature_map(params, x)
    plt.scatter(x, yhat, label="Adjusted")
    plt.scatter(x, x, label="Initial")
    plt.legend(frameon=False)
    plt.xlabel("Inputs")
    plt.title("Inputs", loc="left")
    plt.show()
