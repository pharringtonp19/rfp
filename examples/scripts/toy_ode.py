from diffrax import diffeqsolve, ODETerm, Dopri5
import jax.numpy as jnp
import jax 
from rfp import pjit_time_grad

def f(t, y, args):
    return -y

term = ODETerm(f)
solver = Dopri5()
y0 = jax.random.normal(jax.random.PRNGKey(0), shape=(8, 2))

def f(y0):  
    solution = diffeqsolve(term, solver, t0=0, t1=1, dt0=0.1, y0=y0).ys
    return solution

print(jax.vmap(f)(y0).shape)
pjit_time_grad(f, y0)