import jax 
import jax.numpy as jnp 

def gaussian_kernel(x, z):
    """The Gaussian Kernel"""
    diff = x - z 
    exponent = -(1/2)*jnp.linalg.norm(diff)**2
    return jnp.exp(exponent)

def laplace_kernel(x, z):
    """The Laplace Kernel"""
    diff = x-z 
    exponent = -jnp.linalg.norm(diff)
    return jnp.exp(exponent)

def linear_kernel(x, z):
    """The Linear Kernel"""
    return jnp.dot(x, z)

def your_kernel(phi, x, z):
    """It's Your Kernel!"""
    x_mapped = phi(x)
    z_mapped = phi(z)
    return jnp.dot(x_mapped, z_mapped)

def local_weight(kernel, x, X):
    """Weights assigned to the training points with
    respect to some value `x`"""
    return jax.vmap(kernel, in_axes=(None, 0))(x, X)

def batch_weights(kernel, X):
    lw = lambda x : local_weight(kernel, x, X)
    return jax.vmap(lw)(X)

def opt_weights(kernel, data):
    Y, X = data 
    K = batch_weights(kernel, X)
    w_opt = jnp.linalg.inv(K) @ Y 
    return w_opt 

def predict(kernel, w, x, X):
    lw = local_weight(kernel, x, X)
    return lw @ w

def batch_predict(kernel, w, X):
    K = batch_weights(kernel, X)
    return K @ w
    
def batch_loss(kernel, w, data):
    Y, X = data 
    Yhat = batch_predict(kernel, w, X)
    return jnp.mean((Y-Yhat)**2)



if __name__ == '__main__':
    import matplotlib.pyplot as plt 
    import jax 
    from functools import partial 

    kernel_type = gaussian_kernel
    X = jax.random.ball(jax.random.PRNGKey(0), d=10, p=2, shape=(100,))
    Y = jax.random.normal(jax.random.PRNGKey(1), shape=(100,1))
    w_opt = opt_weights(kernel_type, (Y, X))
    print(batch_loss(kernel_type, w_opt, (Y, X)))
 