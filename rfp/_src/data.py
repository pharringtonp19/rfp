"""All data files should be placed in here"""


import jax
import jax.numpy as jnp

from rfp._src.types import Data


def f1(x):
    return jnp.log(x**2 + 1.0 + jnp.sin(x * 1.5)) + 1.5


def sample1(continuous, f, key, features: int) -> Data:
    subkey1, subkey2, subkey3 = jax.random.split(key, 3)
    if continuous:
        treatment = jax.random.normal(subkey1)
    else:
        treatment = jax.random.bernoulli(subkey1).astype(jnp.float32)
    covariates = jax.random.normal(subkey2, shape=(features,))
    outcome = f(treatment) + jax.random.normal(subkey3)
    return outcome, treatment, covariates


def sample2(key, n: int) -> Data:
    xs = jnp.linspace(-1.5, 1.5, n).reshape(-1, 1)
    ys = xs**3
    return (ys, xs)


def sample3(key, features: int) -> Data:
    subkey1, subkey2, subkey3, subkey4 = jax.random.split(key, 4)
    treatment = jax.random.normal(subkey1)
    time = jax.random.bernoulli(subkey2).astype(jnp.float32)
    covariates = jax.random.normal(subkey3, shape=(features,))
    outcome = (
        1 * time + 1 * treatment + 2 * treatment * time + jax.random.normal(subkey4)
    )
    return outcome, treatment, time, covariates


def sample4(key, scales) -> Data:
    """Work on This!"""
    subkey1, subkey2 = jax.random.split(key, 2)
    treatment = jax.random.bernoulli(subkey1).astype(jnp.float32)
    rate = jax.random.uniform(subkey2, minval=0.0, maxval=1.0)
    outcome = jnp.sin(rate * 10)
    return outcome, jnp.ones_like(outcome), rate, treatment


# ================================================
#               DOUBLE MACHINE LEARNING
# ================================================
from scipy.linalg import cholesky, toeplitz


def VC2015(key, theta, n_obs, features):
    a0 = 1
    a1 = 0.25
    s1 = 1
    b0 = 1
    b1 = 0.25
    s2 = 1
    subkey1, subkey2, subkey3 = jax.random.split(key, 3)
    vec = [0.7 ** (i) for i in range(features)]
    S = cholesky(toeplitz(vec))
    X = jnp.dot(jax.random.normal(subkey1, (n_obs, features)), S)
    z = jnp.exp(X)
    D = (
        a0 * X[:, 1].reshape(-1, 1)
        + a1 * jnp.exp(X[:, 3]).reshape(-1, 1) / (1 + jnp.exp(X[:, 3])).reshape(-1, 1)
        + jax.random.normal(subkey2, (n_obs, 1))
    )
    G = b0 * jnp.exp(X[:, 1]).reshape(-1, 1) / (1 + jnp.exp(X[:, 1])).reshape(
        -1, 1
    ) + b1 * X[:, 3].reshape(-1, 1)
    Y = theta * D + G + s2 * jax.random.normal(subkey3, (n_obs, 1))
    return Y, D, X


if __name__ == "__main__":
    s = sample3(jax.random.PRNGKey(0), 2)
    print(s)
