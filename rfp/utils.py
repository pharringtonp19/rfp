import jax 

def batch_sample(sample_fn, key, n, d):
    """Samples Batch of Data"""
    D, X, Y = jax.vmap(sample_fn, in_axes=(0, None))(jax.random.split(key, n), d)
    return D.reshape(-1,1), X, Y.reshape(-1,1)

def init_keys(key_num):
    """Data and Params Keys"""
    return jax.random.split(jax.random.PRNGKey(key_num))