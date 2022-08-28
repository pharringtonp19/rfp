import os
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

github_folder = str(Path(os.getcwd()).parent.absolute())
file_link: str = github_folder + "/eviction_paper/limitations_ols/"
print(file_link)


def sample(key, n, p, effect):
    subkey1, subkey2 = jax.random.split(key, 2)
    D = jax.random.bernoulli(subkey1, p, shape=(n, 1))
    Y = effect * D + 0.1 * jax.random.normal(subkey2, shape=(n, 1))
    return D, Y


def batch_sample(key, p, effect, n):
    ds, ys = sample(key, n, p, effect)
    return ds, ys


def get_coeff(ds, ys):
    return jnp.linalg.lstsq(ds - jnp.mean(ds), ys)[0][0]


def estimate(key, p1, p2, e1, e2, n):
    subkey1, subkey2 = jax.random.split(key)
    ds1, ys1 = batch_sample(subkey1, p1, e1, n)
    coef1 = get_coeff(ds1, ys1)
    ds2, ys2 = batch_sample(subkey2, p2, e2, n)
    coef2 = get_coeff(ds2, ys2)
    return 0.5 * coef1 + 0.5 * coef2


def simulate(p):
    return jax.vmap(estimate, in_axes=(0, None, None, None, None, None))(
        jax.random.split(jax.random.PRNGKey(0), 200), 0.5, p, 1.0, 3.0, 1000
    )


def f(p):
    ans = simulate(p)
    return jnp.mean(ans), jnp.std(ans)


ps = jnp.linspace(0.1, 0.9, 50)
upper_results = []
mean_results = []
lower_results = []
for i in ps:
    ans = f(i)
    upper_results.append(ans[0] + 2 * ans[1])
    mean_results.append(ans[0])
    lower_results.append(ans[0] - 2 * ans[1])

fig = plt.figure(dpi=300, tight_layout=True)
plt.plot(ps, mean_results, label=r"$\hat{\beta}$")
plt.plot(ps, upper_results, label=r"$\hat{\beta} + 2\sigma_{\hat{\beta}}$")
plt.plot(ps, lower_results, label=r"$\hat{\beta} - 2\sigma_{\hat{\beta}}$")
plt.legend(frameon=False)
plt.title("Estimate", loc="left", size=14)
plt.xlabel("Propensity Score", size=14)
plt.ylim(1.4, 2.1)
filename = file_link + "saturated_cluster.png"
fig.savefig(filename, format="png")
plt.show()
