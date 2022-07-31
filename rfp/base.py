import chex 
from typing import Tuple 

Array = chex.Array # Is this correct?
OptState = chex.ArrayTree   
Params = chex.ArrayTree 
Data = Tuple[Array, Array] | Tuple[Array, Array, Array]
Key  = chex.PRNGKey
