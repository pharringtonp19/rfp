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
        
        # Initialize Model Parameters
        body = mlp.init_fn(key, features)
        
        # He initialization for the head and bias
        stddev_head = jnp.sqrt(2.0 / fwd_pass_layer)
        head = stddev_head * jax.random.normal(k1, (fwd_pass_layer, head_dim))
        
        stddev_bias = jnp.sqrt(2.0 / 1)
        bias = stddev_bias * jax.random.normal(k2, (1, head_dim))
        
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
