import jax
from chex import assert_shape

"""
- A decorator is just syntactic sugar for a partially evaluated higher order function?"""


def batch_sample_time(n):
    def decorator(sampler):
        def wrapper(key, features):
            Y, D, T, X = jax.vmap(sampler, in_axes=(0, None))(
                jax.random.split(key, n), features
            )
            Y, D, T = Y.reshape(-1, 1), D.reshape(-1, 1), T.reshape(-1, 1)
            assert_shape([Y, D, T, X], [(n, 1), (n, 1), (n, 1), (n, features)])
            return Y, D, T, X

        return wrapper

    return decorator
