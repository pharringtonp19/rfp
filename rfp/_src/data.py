import jax 
import jax.numpy as jnp 
from rfp._src.types import Data

def f1(x):
    return jnp.log(x**2 + 1.0 + jnp.sin(x * 1.5)) + 1.5

def sample1(f, key, features: int) -> Data:
    subkey1, subkey2, subkey3 = jax.random.split(key, 3)
    treatment = jax.random.normal(subkey1)
    covariates = jax.random.normal(subkey2, shape=(features,))
    outcome = f(treatment) + jax.random.normal(subkey3)
    return outcome, treatment, covariates

def sample2(key, n: int) -> Data: 
    xs = jnp.linspace(-1.5, 1.5, n).reshape(-1,1)
    ys = xs**3 
    return (ys, xs)

def sample3(key, features: int) -> Data:
    subkey1, subkey2, subkey3, subkey4 = jax.random.split(key, 4)
    treatment = jax.random.normal(subkey1)
    time = jax.random.bernoulli(subkey2).astype(jnp.float32)
    covariates = jax.random.normal(subkey3, shape=(features,))
    outcome = 1*time + 1*treatment + 2*treatment*time + jax.random.normal(subkey4)
    return outcome, treatment, time, covariates


if __name__ == "__main__":
    s = sample3(jax.random.PRNGKey(0), 2)
    print(s)