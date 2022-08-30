import jax 
import jax.numpy as jnp 
from jax.experimental import maps 
from jax.experimental import PartitionSpec 
from jax.experimental.pjit import pjit 
import numpy as np 

def pjit_key(sims):

    def pjit_decorator(fn):
        mesh_shape = (4,)
        devices = np.array(jax.devices()).reshape(*mesh_shape)
        mesh = maps.Mesh(devices, ('x',))
        print(mesh)

        f = pjit(
        lambda keys: jax.vmap(fn)(keys),
        in_axis_resources=PartitionSpec('x',),
        out_axis_resources=PartitionSpec('x',))

        def wrapper(key):
            with maps.Mesh(mesh.devices, mesh.axis_names):
                data = f(jax.random.split(key, sims))
            return data
        return wrapper 

    return pjit_decorator

    
 
    # Sends data to accelerators based on partition_spec
