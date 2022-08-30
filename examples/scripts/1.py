import jax 
import jax.numpy as jnp 
from jax.experimental import maps 
from jax.experimental import PartitionSpec 
from jax.experimental.pjit import pjit 
import numpy as np 
from rfp import MLP 
from functools import partial 

mlp = MLP([32, 32, 1])
params = mlp.init_fn(jax.random.PRNGKey(0), 2)

mesh_shape = (4,)
devices = np.array(jax.devices()).reshape(*mesh_shape)
mesh = maps.Mesh(devices, ('x',))
print(mesh)

input_data = np.arange(8 * 2).reshape(8, 2).astype(jnp.float32)

@jax.grad
def loss_fn(params, data):
    yhat = mlp.fwd_pass(params, data)
    return jnp.reshape(yhat**2, ())

f = pjit(
  lambda data: jax.tree_util.tree_map(partial(loss_fn, params), data),
  in_axis_resources=PartitionSpec('x',),
  out_axis_resources=PartitionSpec('x',))
 
#Sends data to accelerators based on partition_spec
with maps.Mesh(mesh.devices, mesh.axis_names):
    data = f(input_data)
print(type(data))
# for i in data.device_buffers:
#     print(i.shape)
# z = jax.vmap(partial(mlp.fwd_pass, params))(input_data)
# print(z)