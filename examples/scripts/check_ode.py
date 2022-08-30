import jax 
import jax.numpy as jnp 
from jax.experimental import maps 
from jax.experimental import PartitionSpec 
from jax.experimental.pjit import pjit 
import numpy as np 
from diffrax import diffeqsolve, ODETerm, Dopri5


print(jax.devices())
print(jax.devices('cpu'))
mesh_shape = (4,)
devices = np.array(jax.devices()).reshape(*mesh_shape)
mesh = maps.Mesh(devices, ('x',))
print(mesh)

def vector_field(t, y, args):
    return -y

term = ODETerm(vector_field)
solver = Dopri5()
input_data = np.arange(8 * 2).reshape(8, 2).astype(jnp.float32)

def ivp(y0):  
    solution = diffeqsolve(term, solver, t0=0, t1=1, dt0=0.1, y0=y0).ys
    return solution

print(jax.vmap(ivp)(input_data).shape)

f = pjit(
  lambda data: jax.vmap(ivp)(data),
  in_axis_resources=PartitionSpec('x',),
  out_axis_resources=PartitionSpec('x',))
 
#Sends data to accelerators based on partition_spec
with maps.Mesh(mesh.devices, mesh.axis_names):
    data = f(input_data)
print(type(data))

