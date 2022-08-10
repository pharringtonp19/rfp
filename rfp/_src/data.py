import jax 
import jax.numpy as jnp 
from _src.base import Data 

def sample1(f, key, features: int) -> Data:
    subkey1, subkey2, subkey3 = jax.random.split(key, 3)
    treatment = jax.random.normal(subkey1)
    covariates = jax.random.normal(subkey2, shape=(features,))
    outcome = f(treatment) + jax.random.normal(subkey3)
    return treatment, covariates, outcome

def sample2(key, n: int) -> Data: 
    xs = jnp.linspace(-1.5, 1.5, n).reshape(-1,1)
    ys = xs**3 
    return (ys, xs)