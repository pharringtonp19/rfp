import jax
import jax.numpy as jnp
from jax.experimental import mesh_utils
from jax.sharding import PositionalSharding

sharding = PositionalSharding(mesh_utils.create_device_mesh((4,)))
x = jax.random.normal(jax.random.PRNGKey(0), (8192, 8192))
x = jax.device_put(x, sharding.reshape(2, 2))

@jax.jit
def f(x):
  x = x + 1
  y = jax.lax.with_sharding_constraint(x, sharding.replicate())
  return y


jax.debug.visualize_array_sharding(x)
y = f(x)
jax.debug.visualize_array_sharding(y)
print(x.shape)
print(y.shape)

# def predict(params, inputs):
#   for W, b in params:
#     outputs = jnp.dot(inputs, W) + b
#     inputs = jnp.maximum(outputs, 0)
#   return outputs

# def loss(params, batch):
#   inputs, targets = batch
#   predictions = predict(params, inputs)
#   return jnp.mean(jnp.sum((predictions - targets)**2, axis=-1))

# loss_jit = jax.jit(loss)
# gradfun = jax.jit(jax.grad(loss))


# def init_layer(key, n_in, n_out):
#     k1, k2 = jax.random.split(key)
#     W = jax.random.normal(k1, (n_in, n_out)) / jnp.sqrt(n_in)
#     b = jax.random.normal(k2, (n_out,))
#     return W, b

# def init_model(key, layer_sizes, batch_size):
#     key, *keys = jax.random.split(key, len(layer_sizes))
#     params = list(map(init_layer, keys, layer_sizes[:-1], layer_sizes[1:]))

#     key, *keys = jax.random.split(key, 3)
#     inputs = jax.random.normal(keys[0], (batch_size, layer_sizes[0]))
#     targets = jax.random.normal(keys[1], (batch_size, layer_sizes[-1]))

#     return params, (inputs, targets)

# layer_sizes = [784, 8192, 8192, 8192, 10]
# batch_size = 8192

# params, batch = init_model(jax.random.PRNGKey(0), layer_sizes, batch_size) # WHERE ARE THESE LOCATED
# devices = mesh_utils.create_device_mesh((4,))
# sharding = PositionalSharding(devices).reshape(4, 1)
# batch = jax.device_put(batch, sharding)
# #params = jax.device_put(params, sharding.replicate())
# print(loss_jit(params, batch))

# step_size = 1e-5

# for _ in range(30):
#   grads = gradfun(params, batch)
#   params = [(W - step_size * dW, b - step_size * db)
#             for (W, b), (dW, db) in zip(params, grads)]

# print(loss_jit(params, batch))



# sharding = PositionalSharding(devices)
# sharding = sharding.reshape(2, 2)
# print(sharding)
# x = jax.random.normal(jax.random.PRNGKey(0), shape=(12,12))
# jax.debug.visualize_array_sharding(x)
# y = jax.device_put(x, sharding.replicate(axis=0, keepdims=True))
# jax.debug.visualize_array_sharding(y)

