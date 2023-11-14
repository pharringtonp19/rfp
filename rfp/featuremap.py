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

from rfp.nn import MLP  # Is this necessary?

# from rfp._src.types import Array, Kleisi, ODE_Solver, Params


# def predict(binary, feature_map, params, X):
#     yhat = feature_map(params.body, X) @ params.other + params.bias
#     if binary:
#         return jax.nn.sigmoid(yhat)
#     return yhat


@dataclass
class NeuralODE:
    """Neural ODE"""

    mlp: MLP
    solver: Any
    t1: float
    saveat: Any = SaveAt(t1=True)
    adjoint: Any = BacksolveAdjoint()

    def vector_field(self, t, y, args) :
        """mlp parameterized vector field"""
        state = jnp.hstack([y, jnp.array(t)])
        return self.mlp.fwd_pass(args, state)

    def aug_vector_field(self, t, state, args):
        """Augmented mlp parameterized vector field"""
        y, _ = state[:-1], state[-1]
        a = self.vector_field(t, y, args)
        b = jnp.reshape(
            jnp.linalg.norm(jax.jacobian(self.vector_field)(t, y, args)), (1,)
        )
        c = jnp.concatenate((a, b))
        return c

    def solve_ivp(self, params, input) :
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

    def __call__(self, params, X):
        """Batched FWD Pass"""
        sols = jax.vmap(self.solve_ivp, in_axes=(None, 0))(
            params, jnp.hstack((X, jnp.zeros((X.shape[0], 1))))
        ).ys.squeeze()
        phiX = sols[:, :-1]
        regs = sols[:, -1]
        return phiX, jnp.mean(regs)
