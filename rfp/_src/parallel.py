from dataclasses import dataclass
import jax 
import jax.numpy as jnp 
from jax.experimental import maps 
from jax.experimental import PartitionSpec 
from jax.experimental.pjit import pjit 
import numpy as np 
from einops import rearrange 
from functools import partial 

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


def pre_pmap(devices, data):
    return rearrange(data, '(b h) w -> b h w', b=devices)




def pv_map(n_devices):

  def decorator(f):

    def wrapper(params, data):
      data = pre_pmap(n_devices, data)
      t = lambda data: jax.tree_util.tree_map(partial(f, params), data)
      return jnp.mean(jax.pmap(t)(data))

    return wrapper 
  
  return decorator 

    
 
    # Sends data to accelerators based on partition_spec
