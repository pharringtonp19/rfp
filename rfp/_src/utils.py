import timeit
from dataclasses import dataclass
from functools import partial, wraps
from typing import NamedTuple

import flax.struct
import jax
import jax.numpy as jnp
from chex import assert_shape
from einops import reduce

from rfp._src import parallel
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


def batch_sample_weight(n):
    def decorator(sampler):
        def wrapper(key):
            subkey1, subkey2 = jax.random.split(key)
            scale = jax.random.uniform(subkey1, shape=(3,), maxval=0.7)
            print(scale)
            ys, ws, ts, ds = jax.vmap(sampler, in_axes=(0, None))(
                jax.random.split(key, n), scale
            )
            ys, ws, ts, ds = (
                ys.reshape(-1, 1),
                ws.reshape(-1, 1),
                ts.reshape(-1, 1),
                ds.reshape(-1, 1),
            )
            assert_shape([ys, ws, ts, ds], [(n, 1), (n, 1), (n, 1), (n, 1)])
            return ys, ws, ts, ds

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


def time_grad_pvmap(loss_fn, params, data):
    pv_mapped_loss_fn = parallel.pv_map(4)(loss_fn)
    trials = timeit.repeat(
        stmt=lambda: pv_mapped_loss_fn(params, data), number=1, repeat=2
    )
    print(
        f"Vectorized:\t\t   Compile Time: {trials[0]:.4f} | Compiled Run Time: {trials[1]:.4f}  | Ratio: {trials[0] / trials[1]:.4f}"
    )


def time_grad(loss_fn, params, data):

    jitted_grad_loss_fn = jax.jit(
        jax.grad(loss_fn.__call__, has_aux=loss_fn.aux_status)
    )
    trials = timeit.repeat(
        stmt=lambda: jitted_grad_loss_fn(params, data), number=1, repeat=2
    )
    print(
        f"Standard:\t\t   Compile Time: {trials[0]:.4f} | Compiled Run Time: {trials[1]:.4f}  | Ratio: {trials[0] / trials[1]:.4f}"
    )


# def time_grad(loss_fn, params, data):
#     jitted_grad_loss_fn = jax.jit(
#         jax.grad(loss_fn.__call__, has_aux=loss_fn.aux_status)
#     )
#     trials = timeit.repeat(
#         stmt=lambda: jitted_grad_loss_fn(params, data), number=1, repeat=2
#     )
#     print(
#         f"Compile Time: {trials[0]:.4f} | Compiled Run Time: {trials[1]:.4f}  | Ratio: {trials[0] / trials[1]:.4f}"
#     )


# def pjit_time_grad(f, data):
#     import jax
#     import numpy as np
#     from jax.experimental import PartitionSpec, maps
#     from jax.experimental.pjit import pjit

#     mesh_shape = (4,)  # This is hardcoded atm
#     devices = np.asarray(jax.devices()).reshape(*mesh_shape)
#     mesh = maps.Mesh(devices, ("x",))
#     print(devices)
#     f = pjit(f, in_axis_resources=PartitionSpec("x"), out_axis_resources=None)

#     # Sends data to accelerators based on partition_spec
#     with maps.Mesh(mesh.devices, mesh.axis_names):
#         jax.debug.breakpoint()
#         loss = f(data)
#     print(type(loss))
# y = jnp.mean(loss)
# print(y, y.shape)
# print(loss.shape)
# for i in loss.device_buffers:
#     print(i.shape)
# print(len(loss.device_buffers))


def batchify(func):
    def wrapper(self, params, data):
        cluster_losses = jax.tree_util.tree_map(partial(func, params), data)
        loss = (1 / (len(data))) * jax.tree_util.tree_reduce(
            lambda a, b: a + b, cluster_losses
        )
        return loss

    return wrapper


def split(data):
    Y = data[:, 0].reshape(-1, 1)
    D = data[:, 1].reshape(-1, 1)
    T = data[:, 2].reshape(-1, 1)
    X = data[:, 3:].reshape(data.shape[0], -1)
    return Y, D, T, X


def split_weight(data):
    """
    Example Outcome: Change in eviction fillings
    Model: Bi-level FFWD"""
    ys = data[:, 0].reshape(-1, 1)
    ws = data[:, 1].reshape(-1, 1)
    ts = data[:, 2].reshape(-1, 1)
    return ys, ws, ts


class Model_Params(NamedTuple):
    body: Params
    other: Params
    bias: Params


def init_ode1_model(key, mlp):
    subkey1, subkey2 = jax.random.split(key, 2)
    other = jax.random.normal(subkey1, shape=(1,))
    body = mlp.init_fn(subkey2, 2)
    return Model_Params(body, other)


def store_time_results(path, n, text):
    """Taken from https://stackoverflow.com/questions/4719438/editing-specific-line-in-text-file-in-python"""
    with open(path, "r") as file:
        # read a list of lines into data
        data = file.readlines()
        print(data)

        # now change the 2nd line, note that you have to add a newline
        data[n] = text

        # and write everything back
        with open(path, "w") as file:
            file.writelines(data)


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
