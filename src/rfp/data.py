import jax 
import jax.numpy as jnp 

def sample1(f, key, features):
    subkey1, subkey2, subkey3 = jax.random.split(key, 3)
    treatment = jax.random.normal(subkey1)
    covariates = jax.random.normal(subkey2, shape=(features,))
    outcome = f(treatment) + jax.random.normal(subkey3)
    return treatment, covariates, outcome