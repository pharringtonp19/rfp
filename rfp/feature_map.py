"""Hey"""

import chex
import diffrax
import jax
import jax.numpy as jnp
from diffrax import BacksolveAdjoint, diffeqsolve, ODETerm, PIDController
from flax.core import unfreeze
from dataclasses import dataclass 
from rfp.base import ODE_Solver, Params, Array, Kleisi
from rfp.nn import MLP # Is this necessary?


@dataclass 
class neuralODE: 
    mlp: MLP = MLP([32, 1])
    solver: ODE_Solver = diffrax.Heun()
    t1: float = 1.0 

    def vector_field(self, t, y, args):
        """mlp parameterized vector field"""
        state = jnp.hstack([y, jnp.array(t)])
        return self.mlp.fwd_pass(args, state)
    
    def aug_vector_field(self, t, state, args):
        """augmented mlp parameterized vector field"""
        y, _ = state[:-1], state[-1]
        a = self.vector_field(t, y, args)
        b = jnp.reshape(jnp.linalg.norm(jax.jacobian(self.vector_field)(t, y, args)), (1,))
        c = jnp.concatenate((a, b))
        return c

    def solve_ivp(self, params, input):
        """fwd pass of ode"""
            return diffeqsolve(
                ODETerm(self.aug_vector_field),
                self.solver,
                t0=0,
                t1=self.t1,
                y0=input,
                dt0=None,
                adjoint=BacksolveAdjoint(),
                stepsize_controller=PIDController(rtol=1e-3, atol=1e-3),
                args=params,
            ).ys

    def __call__(self, params: Params, X: Array) -> Kleisi[Array]:
        """batched fwd_pass + regularizer"""
        sols = jax.vmap(self.solve_ivp, in_axes=(None, 0))(
            params, jnp.hstack((X, jnp.zeros((X.shape[0], 1))))
        ).squeeze()
        phiX = sols[:, :-1]
        regs = sols[:, -1]
        return phiX, jnp.mean(regs)

if __name__ == '__main__':
    from rfp.nn import MLP 
    mlp = MLP([32, 1])
    params = mlp.init_fn(jax.random.PRNGKey(0), 2)
    x = jnp.linspace(-3,3,10).reshape(-1,1)
    feature_map = neuralODE()
    yhat, regs = feature_map(params, x)
    print(type(yhat), yhat.shape, type(regs), regs.shape)