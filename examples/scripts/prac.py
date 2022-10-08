import jax
import jax.numpy as jnp
import numpy as np


def f(x, y):
    return 2 * x * y


x, y = 3, 4
i32_scalar = jax.ShapeDtypeStruct((), jnp.dtype("int32"))
f_something = jax.jit(f).lower(i32_scalar, i32_scalar).compile()
print(type(f_something))
