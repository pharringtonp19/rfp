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
    Y = effect * D + jax.random.normal(subkey2, shape=(n, 1))
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

    ds = jnp.vstack((ds1 - jnp.mean(ds1), (ds2 - jnp.mean(ds2))))
    ys = jnp.vstack((ys1, ys2))
    coeff_fe = jnp.linalg.lstsq(ds, ys)[0][0]

    ds = jnp.vstack((ds1, ds2))
    regs = jnp.hstack(
        (jnp.ones_like(ds), ds)
    )  # jnp.vstack((ds1-jnp.mean(ds1), (ds2-jnp.mean(ds2))))
    ys = jnp.vstack((ys1, ys2))
    coeff_no_controls = jnp.linalg.lstsq(regs, ys)[0][0]

    return 0.5 * coef1 + 0.5 * coef2, coeff_fe, coeff_no_controls


def simulate(p):
    return jax.vmap(estimate, in_axes=(0, None, None, None, None, None))(
        jax.random.split(jax.random.PRNGKey(0), 200), 0.1, p, 1.0, 1.0, 100
    )


def f(p):
    ans1, ans2, ans3 = simulate(p)
    return jnp.std(ans1), jnp.std(ans2), jnp.std(ans3)


ps = jnp.linspace(0.1, 0.9, 50)
results1 = []
results2 = []
results3 = []
for i in ps:
    ans = f(i)
    results1.append(ans[0])
    results2.append(ans[1])
    results3.append(ans[2])

fig = plt.figure(dpi=300, tight_layout=True)
plt.plot(ps, results1, label="Fully Saturated Model", color="black")
plt.plot(ps, results2, label="Fixed Effects Model", color="gray")
plt.plot(ps, results3, label="No Controls Model", color="blue")
plt.xlabel("Propensity Score", size=14)
plt.title(r"$\sigma(\hat{\beta})$", loc="left", size=14)
plt.legend(frameon=False, title="Model")
plt.ylim(0, 0.4)
filename = file_link + "cluster_var.png"
fig.savefig(filename, format="png")
plt.show()
