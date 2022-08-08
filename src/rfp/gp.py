class Point: 
    def __init__(self, x: float, y: float):
        self._x = x 
        self._y = y 
    
    @property 
    def x(self):
        return self._x 
    
    @property 
    def y(self):
        return self._y 

point = Point(12, 5)
print(point.x)
print(point.y)




# import jax 
# from jax.config import config
# config.update("jax_enable_x64", True)
# import jax.numpy as jnp 

# # def kernel(x1, x2, scale=10, z=0.25):
# #     t = -(1/2)*(x1-x2)**2*(1/z**2)
# #     return scale*jnp.exp(t)

# def kernel(x1, x2):
#     return jnp.minimum(x1, x2)

# # def kernel(x1, x2):
# #     return x1*x2

# def kernel_matrix(x):
#     t = lambda x0: jax.vmap(kernel, in_axes=(None, 0))(x0, x)
#     return jax.vmap(t)(x)

# if __name__ == '__main__':
#     import distrax 
#     import matplotlib.pyplot as plt
#     k = 10
#     x = jnp.linspace(0, 5, k)
#     mu = jnp.zeros_like(x)
#     sigma = kernel_matrix(x)
#     if k == 5:
#         print(sigma)
#     dist = distrax.MultivariateNormalFullCovariance(mu, sigma)
#     z = dist.sample(seed=jax.random.PRNGKey(0), sample_shape=(4,))
#     # print(z)
#     for i in z:
#         plt.plot(x, i)
#     plt.show()
