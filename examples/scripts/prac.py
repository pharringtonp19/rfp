import jax
from jax.experimental import checkify


def f(x):
    return x**2


checked_grad_f = checkify.checkify(jax.grad(f), errors=checkify.float_checks)


def train1():
    def update(carry, t):
        x = carry
        err, grad = checked_grad_f(x)
        x_new = x - 0.01 * grad
        return x_new, 0.0

    x, vals = jax.lax.scan(update, 3.0, xs=None, length=10)
    return x


print(train1())
