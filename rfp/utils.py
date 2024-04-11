from typing import NamedTuple
import jax
import jax.numpy as jnp
import numpy as np 
from jax.tree_util import register_pytree_node_class

@register_pytree_node_class
class ModelParams(NamedTuple):
    body: dict # Feature Map Parameters
    head: jnp.array # Linear Model Parameters
    bias: jnp.array # Linear Model Bias

    @staticmethod
    def init_fn(key, mlp, features, head_dim=1):
        k1, k2 = jax.random.split(key)
        fwd_pass_layer = mlp.nodes[-1]
        """Initialize Model Parameters"""
        body = mlp.init_fn(key, features)
        head = jax.random.normal(k1, (fwd_pass_layer, head_dim)) ### THIS NEEDS TO BE CHECKED
        bias = jax.random.normal(k2, (1, head_dim))    ### THIS NEEDS TO BE CHECKED
        return ModelParams(body, head, bias)
    
    def __repr__(self) -> str:
        return f"ModelParams(body={self.body}, head={self.head}, bias={self.bias})"
    
    def tree_flatten(self):
        children = (self.body, self.head, self.bias)
        aux_data = None
        return children, aux_data
    
    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children)
    
    @property # This is to work with `jax.debug.visualize_array_sharding'
    def shape(self):
        """A property to return a representative shape."""
        # Example: return the shape of 'head' as it might be the primary parameter
        return self.head.shape

    



def batch_matrix_with_padding(matrix: np.array, zip_codes: np.array) -> dict:
    # Create batches as before
    Batches = {}
    Masks = {}
    unique_zip_codes = np.unique(zip_codes)
    for zip_code in unique_zip_codes:
        indices = np.where(zip_codes == zip_code)[0]
        batch = matrix[indices, :]
        Batches[zip_code] = batch

    # Identify maximum size
    max_rows = max(batch.shape[0] for batch in Batches.values())

    # Pad batches and create masks
    for zip_code, batch in Batches.items():
        padding_rows = max_rows - batch.shape[0]
        
        # Add padding
        padded_batch = np.pad(batch, pad_width=((0, padding_rows), (0, 0)), mode='constant')
        
        # Create mask where 1 indicates original data and 0 indicates padding
        mask = np.pad(np.ones(shape=(batch.shape[0], 1)),pad_width=((0, padding_rows), (0, 0)), mode='constant')
        
        Batches[zip_code] = padded_batch
        Masks[zip_code] = mask
    
    # Convert the dictionary values to a list of matrices
    return np.stack(list(Batches.values())), np.stack(list(Masks.values()))
    

def unpad_matrix(batch_X: np.array, batch_D: np.array, mask: np.array):
    """
    Removes padding from batch matrices based on a given mask.

    Args:
    batch_X (np.array): The first batch matrix with padding.
    batch_D (np.array): The second batch matrix with padding.
    mask (np.array): A mask indicating the original data (non-padded parts).

    Returns:
    tuple: A tuple of two arrays (unpad_X, unpad_D) with padding removed.
    """

    # Validate mask dimensions
    if mask.ndim != batch_X.ndim:
        raise ValueError("Mask and input batch dimensions do not match")

    # Use broadcasting to apply mask and select non-zero rows in one step
    unpad_X = batch_X[mask.astype(bool).reshape(-1)]
    unpad_D = batch_D[mask.astype(bool).reshape(-1)]

    return unpad_X, unpad_D

def unpad_all_matrices(X: np.array, D: np.array, masks: np.array) -> dict:
    original_X = {}
    original_D = {}

    for i in range(X.shape[0]):
        original_X[i], original_D[i] = unpad_matrix(X[i], D[i], masks[i])
        original_X[i], original_D[i] = original_X[i].reshape(-1, X.shape[-1]), original_D[i].reshape(-1, D.shape[-1])
    
    return np.vstack((list(original_X.values()))), np.vstack((list(original_D.values())))


# def compute_cost_analysis(f):
#     def partial_eval_func(*args, **kwargs):
#         lowered = jax.jit(f).lower(*args, **kwargs)
#         compiled = lowered.compile()
#         return (
#             compiled.cost_analysis()[0]["flops"],
#             compiled.cost_analysis()[0]["bytes accessed"],
#         )

#     return partial_eval_func


# def einops_reduce(str1: str, str2: str) -> callable:
#     def decorator(f: callable) -> callable:
#         def wrapper(*args, **kwargs):
#             vals = f(*args, **kwargs)
#             return jnp.reshape(reduce(vals, str1, str2), ())

#         return wrapper

#     return decorator


# """
# - A decorator is just syntactic sugar for a partially evaluated higher order function?"""


# def batch_sample_weight(n):
#     def decorator(sampler):
#         def wrapper(key):
#             subkey1, subkey2 = jax.random.split(key)
#             scale = jax.random.uniform(subkey1, shape=(3,), maxval=0.7)
#             print(scale)
#             ys, ws, ts, ds = jax.vmap(sampler, in_axes=(0, None))(
#                 jax.random.split(key, n), scale
#             )
#             ys, ws, ts, ds = (
#                 ys.reshape(-1, 1),
#                 ws.reshape(-1, 1),
#                 ts.reshape(-1, 1),
#                 ds.reshape(-1, 1),
#             )
#             assert_shape([ys, ws, ts, ds], [(n, 1), (n, 1), (n, 1), (n, 1)])
#             return ys, ws, ts, ds

#         return wrapper

#     return decorator


# def training_sampler(batch_size, data, *, key):
#     """
#     Instead of greating a data loader as a generator as in https://docs.kidger.site/equinox/examples/train_rnn/
#     during the training loop, we scan over keys and draw samples of `batch_size`

#     - For supervised training, we want to subsample observations
#     - For cluster training, though, we want to subsample clusters
#     """
#     if batch_size == data[0].shape[0]:
#         return data
#     else:
#         perm = jax.random.permutation(key, data[0].shape[0])
#         batch_perm = perm[:batch_size]
#         sample = tuple(array[batch_perm] for array in data)
#         return sample


# def time_grad_pvmap(loss_fn, params, data):
#     pv_mapped_loss_fn = parallel.pv_map(4)(loss_fn)
#     trials = timeit.repeat(
#         stmt=lambda: pv_mapped_loss_fn(params, data), number=1, repeat=2
#     )
#     print(
#         f"Vectorized:\t\t   Compile Time: {trials[0]:.4f} | Compiled Run Time: {trials[1]:.4f}  | Ratio: {trials[0] / trials[1]:.4f}"
#     )


# def time_grad(loss_fn, params, data):

#     jitted_grad_loss_fn = jax.jit(
#         jax.grad(loss_fn.__call__, has_aux=loss_fn.aux_status)
#     )
#     trials = timeit.repeat(
#         stmt=lambda: jitted_grad_loss_fn(params, data), number=1, repeat=2
#     )
#     print(
#         f"Standard:\t\t   Compile Time: {trials[0]:.4f} | Compiled Run Time: {trials[1]:.4f}  | Ratio: {trials[0] / trials[1]:.4f}"
#     )


# # def time_grad(loss_fn, params, data):
# #     jitted_grad_loss_fn = jax.jit(
# #         jax.grad(loss_fn.__call__, has_aux=loss_fn.aux_status)
# #     )
# #     trials = timeit.repeat(
# #         stmt=lambda: jitted_grad_loss_fn(params, data), number=1, repeat=2
# #     )
# #     print(
# #         f"Compile Time: {trials[0]:.4f} | Compiled Run Time: {trials[1]:.4f}  | Ratio: {trials[0] / trials[1]:.4f}"
# #     )


# def pjit_time_grad(f, data):
#     from jax.experimental import PartitionSpec, maps
#     from jax.experimental.pjit import pjit

#     mesh_shape = (4,)  # This is hardcoded atm
#     devices = np.asarray(jax.devices()).reshape(*mesh_shape)
#     mesh = maps.Mesh(devices, ("x",))
#     print(devices)
#     f = pjit(f, in_axis_resources=PartitionSpec("x"), out_axis_resources=None)

#     # Sends data to accelerators based on partition_spec
#     with maps.Mesh(mesh.devices, mesh.axis_names):
#         loss = f(data)


# #     print(type(loss))
# # y = jnp.mean(loss)
# # print(y, y.shape)
# # print(loss.shape)
# # for i in loss.device_buffers:
# #     print(i.shape)
# # print(len(loss.device_buffers))


# def batchify(func):
#     def wrapper(self, params, data):
#         cluster_losses = jax.tree_util.tree_map(partial(func, params), data)
#         loss = (1 / (len(data))) * jax.tree_util.tree_reduce(
#             lambda a, b: a + b, cluster_losses
#         )
#         return loss

#     return wrapper


# def split(data):
#     Y = data[:, 0].reshape(-1, 1)
#     X = data[:, 1:]
#     return Y, X


# def split_weight(data):
#     """
#     Example Outcome: Change in eviction fillings
#     Model: Bi-level FFWD"""
#     ys = data[:, 0].reshape(-1, 1)
#     ws = data[:, 1].reshape(-1, 1)
#     ts = data[:, 2].reshape(-1, 1)
#     return ys, ws, ts


# def init_ode1_model(key, mlp):
#     subkey1, subkey2 = jax.random.split(key, 2)
#     other = jax.random.normal(subkey1, shape=(1,))
#     body = mlp.init_fn(subkey2, 2)
#     return Model_Params(body, other)


# def store_time_results(path, n, text):
#     """Taken from https://stackoverflow.com/questions/4719438/editing-specific-line-in-text-file-in-python"""
#     with open(path, "r") as file:
#         # read a list of lines into data
#         data = file.readlines()
#         print(data)

#         # now change the 2nd line, note that you have to add a newline
#         data[n] = text

#         # and write everything back
#         with open(path, "w") as file:
#             file.writelines(data)


# if __name__ == "__main__":
#     """This should be made into a test"""

#     data = (1, 2)
#     split(data)
#     split(jnp.array([1, 2]))
#     # from rfp import sample3

#     # n = 100
#     # batch_size = 32
#     # features = 10
#     # data = batch_sample_time(n)(sample3)(jax.random.PRNGKey(0), features)
#     # sample = batch_sample(batch_size, data, key=jax.random.PRNGKey(1))
#     # for i in sample:
#     #     assert_shape(
#     #         [sample[0], sample[1], sample[2], sample[3]],
#     #         [(batch_size, 1), (batch_size, 1), (batch_size, 1), (batch_size, features)],
#     #     )
