### TLDR

We don't assume cross-sectional selection on observables:

$$ Y_i(d, 1) \perp D_i | X_i  \quad \forall d $$

We assume de-panelled selection[^1] on observables:

$$Y_i(d,1) - Y_i(0, 0) \perp D_i \big | X_i \quad \forall d $$

This assumption provides justification for interpreting $\beta_1$ as the causal effect:

$$\begin{align*} Y_{1,i} - Y_{0,i} &= \beta_0 + \beta_1 D_i +  \varepsilon _i  \\  
Y_{1,i}  &= \beta_0 + \beta_1 D_i + Y_{0,i} +  \varepsilon _i \end{align*} $$

In practice, we don't usually condition on $X_i$ (which means there's no reason to correct for $\mathbb{P}(D|X)$, and this is very attractive because we know that the partially linear model fails to correct for this! --> See [Double Machine Learning Example](./examples.md))
and we allow 

$$Y_{0, i} \approx f(X_i, D_i) $$

But in some sense, this kinda feels like cheating because it suggests that de-panelled selection on observables is easier to work with than cross-sectional selection on observables. 

[^1]: Repeated cross-section