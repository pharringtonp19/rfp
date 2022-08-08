import chex 
from typing import TypeAlias, TypeVar
import diffrax
import jax.numpy as jnp 

Array: TypeAlias = chex.Array # Is this correct?
OptState: TypeAlias = chex.ArrayTree   
Params: TypeAlias = chex.ArrayTree 
Data2: TypeAlias= tuple[Array, Array]
Data3: TypeAlias = tuple[Array, Array, Array]
Data = TypeVar('Data', Data2, Data3)
Key: TypeAlias  = chex.PRNGKey
ODE_Solver: TypeAlias = diffrax.solver.base._MetaAbstractSolver
T = TypeVar("T")
Kleisi: TypeAlias = tuple[T, jnp.float32]
