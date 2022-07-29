from typing import Sequence, TypeAlias, TypeVar

import chex
import diffrax
import jax
import jax.numpy as jnp
from diffrax import BacksolveAdjoint, diffeqsolve, ODETerm, PIDController
from flax.core import unfreeze


# --------------- TYPES --------------------------#

Array: TypeAlias = chex.Array
Data: TypeAlias = tuple[Array, Array, Array]
Params: TypeAlias = chex.ArrayTree
T = TypeVar("T")
Kleisi: TypeAlias = tuple[T, float]



def init_vector_field(mlp):

    # @jax.jit
    def vector_field(t, y, args):
        state = jnp.hstack([y, jnp.array(t)])
        return mlp.apply({"params": args}, state)

    return vector_field


def init_aug_vector_field(mlp):

    vector_field = init_vector_field(mlp)

    # @jax.jit
    def aug_vector_field(t, state, args):
        y, _ = state[:-1], state[-1]
        a = vector_field(t, y, args)
        b = jnp.reshape(jnp.linalg.norm(jax.jacobian(vector_field)(t, y, args)), (1,))
        c = jnp.concatenate((a, b))
        return c

    return aug_vector_field


def init_solve_ivp(mlp, solver, t1):

    term = init_aug_vector_field(mlp)

    # @jax.jit
    def solve_ivp(params, input):
        return diffeqsolve(
            ODETerm(term),
            solver,
            t0=0,
            t1=t1,
            y0=input,
            dt0=None,
            adjoint=BacksolveAdjoint(),
            stepsize_controller=PIDController(rtol=1e-3, atol=1e-3),
            args=params,
        ).ys

    return solve_ivp


def init_neuaral_ode(mlp: MLP, solver: diffrax.solver, t1: float):

    solve_ivp = init_solve_ivp(mlp, solver, t1)

    # @jax.jit
    def neural_ode(params: Params, X: Array) -> Kleisi[Array]:
        sols = jax.vmap(solve_ivp, in_axes=(None, 0))(
            params, jnp.hstack((X, jnp.zeros((X.shape[0], 1))))
        ).squeeze()
        phiX = sols[:, :-1]
        regs = sols[:, -1]
        return phiX, jnp.mean(regs)

    def init_model(key, features):
        params = unfreeze(mlp.init(key, jnp.ones((1, features + 1))))["params"]
        return params

    return neural_ode, init_model


def init_ffwd(mlp: MLP):
    def ffwd(params, X):
        return mlp.apply({"params": params}, X), 0.0

    def init_model(key, features):
        params = unfreeze(mlp.init(key, jnp.ones((1, features))))["params"]
        return params

    return ffwd, init_model
