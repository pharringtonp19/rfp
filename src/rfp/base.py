import chex 
from typing import TypeAlias, TypeVar
import diffrax

Array: TypeAlias = chex.Array # Is this correct?
OptState: TypeAlias = chex.ArrayTree   
Params: TypeAlias = chex.ArrayTree 
Data: TypeAlias = tuple[Array, Array] | tuple[Array, Array, Array]
Key: TypeAlias  = chex.PRNGKey
ODE_Solver: TypeAlias = diffrax.solver.base._MetaAbstractSolver
T = TypeVar("T")
Kleisi: TypeAlias = tuple[T, jnp.float32]
