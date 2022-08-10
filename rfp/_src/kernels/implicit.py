import tree_math as tm 
import jax.numpy as jnp 

v = tm.Vector({'x': 1, 'y': jnp.arange(2, 4)})
print(v)

@tm.struct
class Point: 
    x: float 
    y: float 

# a = Point(0., 1.)
# b = Point(1., 1.)
# c = a + b 

# print(b)
