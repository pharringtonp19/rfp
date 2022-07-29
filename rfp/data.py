import jax 
import jax.numpy as jnp 
import diffrax 


def sample1(key, features):
    subkey1, subkey2, subkey3 = jax.random.split(key, 3)
    f = lambda d: jnp.log(d**2 + 1.0 + jnp.sin(d * 1.5)) + 1.5
    treatment = jax.random.normal(subkey1)
    covariates = jax.random.normal(subkey2, shape=(features,))
    outcome = f(treatment) + jax.random.normal(subkey3)
    return treatment, covariates, outcome