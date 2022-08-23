import timeit
from dataclasses import dataclass
from functools import partial, wraps

import flax.struct
import jax
import jax.numpy as jnp
from chex import assert_shape
from einops import reduce

from rfp._src.types import Params


def einops_reduce(str1: str, str2: str) -> callable:
    def decorator(f: callable) -> callable:
        def wrapper(*args, **kwargs):
            vals = f(*args, **kwargs)
            return jnp.reshape(reduce(vals, str1, str2), ())

        return wrapper

    return decorator


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


def training_sampler(batch_size, data, *, key):
    """
    Instead of greating a data loader as a generator as in https://docs.kidger.site/equinox/examples/train_rnn/
    during the training loop, we scan over keys and draw samples of `batch_size`

    - For supervised training, we want to subsample observations
    - For cluster training, though, we want to subsample clusters
    """
    if batch_size == data[0].shape[0]:
        return data
    else:
        perm = jax.random.permutation(key, data[0].shape[0])
        batch_perm = perm[:batch_size]
        sample = tuple(array[batch_perm] for array in data)
        return sample


def time_grad(loss_fn, params, data):
    jitted_grad_loss_fn = jax.jit(
        jax.grad(loss_fn.__call__, has_aux=loss_fn.aux_status)
    )
    trials = timeit.repeat(
        stmt=lambda: jitted_grad_loss_fn(params, data), number=1, repeat=2
    )
    print(
        f"Compile Time: {trials[0]:.4f} | Run Time: {trials[1]:.4f} | Ratio: {trials[0] / trials[1]:.4f}"
    )


def batchify(func):
    def wrapper(self, params, data):
        cluster_losses = jax.tree_util.tree_map(partial(func, params), data)
        loss = (1 / (len(data))) * jax.tree_util.tree_reduce(
            lambda a, b: a + b, cluster_losses
        )
        return loss

    return wrapper


def split(data):
    """We should include a check for this"""

    if isinstance(data, tuple):
        return data
    else:
        Y = data[:, 0].reshape(-1, 1)
        D = data[:, 1].reshape(-1, 1)
        T = data[:, 2].reshape(-1, 1)
        X = data[:, 3:].reshape(data.shape[0], -1)
        return Y, D, T, X


def Model_Params(params, linear_params=None):
    return {"ode_params": params, "linear_params": linear_params}


if __name__ == "__main__":
    """This should be made into a test"""

    data = (1, 2)
    split(data)
    split(jnp.array([1, 2]))
    # from rfp import sample3

    # n = 100
    # batch_size = 32
    # features = 10
    # data = batch_sample_time(n)(sample3)(jax.random.PRNGKey(0), features)
    # sample = batch_sample(batch_size, data, key=jax.random.PRNGKey(1))
    # for i in sample:
    #     assert_shape(
    #         [sample[0], sample[1], sample[2], sample[3]],
    #         [(batch_size, 1), (batch_size, 1), (batch_size, 1), (batch_size, features)],
    #     )
