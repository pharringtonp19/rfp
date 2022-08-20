from typing import TypeAlias

import chex
import jax
import jax.numpy as jnp


# --------------- TYPES --------------------------#

Array: TypeAlias = chex.Array

def linear_model_time(Y, D, T, X):
    regressors = jnp.hstack((D*T, D, T, jnp.ones_like(D), X))
    residuals = jnp.linalg.lstsq(regressors, Y)[1][0]
    return residuals / X.shape[0], 0.0

def linear_model1(Y, D, X):
    regressors = jnp.hstack((D, jnp.ones_like(D), X))
    residuals = jnp.linalg.lstsq(regressors, Y)[1][0]
    return residuals / X.shape[0], 0.0


def linear_model2(target, X):
    """Linear Model Without Treatment"""
    regressors = jnp.hstack((jnp.ones((X.shape[0], 1)), X))
    residuals = jnp.linalg.lstsq(regressors, target)[1][0]
    return residuals / X.shape[0], 0.0


if __name__ == "__main__":
    Y = jax.random.normal(jax.random.PRNGKey(2), shape=(100, 1))
    D = jax.random.normal(jax.random.PRNGKey(0), shape=(100, 1))
    X = jax.random.normal(jax.random.PRNGKey(1), shape=(100, 10))

    z0, z1 = linear_model2(D, X)
    print(type(z0), type(z1), z0.shape)