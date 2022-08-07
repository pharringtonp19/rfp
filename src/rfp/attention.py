import jax 
import jax.numpy as jnp 
from rfp.base import Array 


def weight(xi: Array, xj: Array) -> Array:
    """Compute Weights"""
    return jnp.dot(xi, xj)

def normalize(z: Array) -> Array: 
    """Softmax Normalization"""
    return jax.nn.softmax(z)

def weight_array(x, X):
    """Compute weights"""
    z = jax.vmap(weight, in_axes=(None,0))(x, X)
    return normalize(z)

def weighted_avg(X, w):
    """Compute Weighted Average"""
    return X @ w 


if __name__ == '__main__':
    import distrax 
    import matplotlib.pyplot as plt
    k = 5
    mu = jnp.zeros(shape=(k,))
    sigma = jnp.ones(shape=(k,))
    dist = distrax.MultivariateNormalDiag(mu, sigma)
    sample = dist.sample(seed=jax.random.PRNGKey(0), sample_shape=(4,))
    for i in sample:
        plt.plot(jnp.arange(k), i)
    plt.show()
    # X = jnp.sort(jax.random.normal(jax.random.PRNGKey(0), shape=(100,)))
    # y = jnp.sin(X)
    # plt.plot(X, y)
    # plt.show()