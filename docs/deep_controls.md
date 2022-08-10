??? tip "What's In Between?"

    - "De-meaning the estimates" as in a partially linear model 
    - Reweighting observations as in a fully nonparametric model


#### **Regression Framework**

- You can think of the regression framework as averaging across $n$ randomized controls trials centered on $x$ with one observation in each trial with no correction/adjustment for the conditional distribution: $\mathbb{P}(D|X)$

$$\begin{align*} 
\hat{\theta} &= \underset{\theta}{\text{argmin}} \ 
\sum _i \big(v_i -\theta u_i)\big)^2\end{align*}$$

where

- $v_i = y_i - f_1(x_i)$
- $u_i= d_i - f_2(x_i)$
- $f_1(x_i) = E[Y|X=x]$
- $f_2(x_i) = E[D|X=x]$

??? note "Causal Inference"

    - Statistical Learning Theory: "If it turns out that nevertheless we can explain the data at hand, then we have reason to believe that we have found a regularity underlying the data."
   

#### **Our Gradient Correction Approach**

$$ \begin{align*}  E[Y|X] &= \beta ^T \phi(X) \\ 
&= \int Y(D)d\mathbb{P}(D|X) \\
E[Y|D,X] &= \theta^TD + E[Y|X] \end{align*} $$


!!! tldr "Overview"

    The previous section highlighted how bi-level gradient descent is a potentially an attactive approach when estimating nonparametrics estimands with clustered data. It may not be immediatley clear, though, why this "gradient based" partialled out approach is also well suited when were intersted in estimating a parametric estimand. 

We incorporate the regularizing strategy proposed in [Learning Differential Equations that are Easy to Solve
](https://arxiv.org/abs/2007.04504)  
<figure markdown>
![Image title](./../fig/reg_ode_0.0.png){ width="700" }
![Image title](./../fig/reg_ode_10.0.png){ width="700" }
<figcaption>Visual Effect of Regularizing</figcaption>
</figure>  

??? tip "Abstract Algebra"

    - Sets
    - Group (functions defined on this set)

    What structure on $\mathbb{R}^n$ should we preserve? 

    $$\begin{align*} g : {\text{bijections}} \to \mathbb{R^n} \to \mathbb{R^n} \end{align*}  \equiv \text{Sym}(\mathbb{R}^n)$$




We don't assume cross-sectional selection on observables:

$$ Y_i(d, 1) \perp D_i | X_i  \quad \forall d $$

We assume de-panelled selection[^1] on observables:

$$Y_i(d,1) - Y_i(0, 0) \perp D_i \big | X_i \quad \forall d $$

This assumption provides justification for interpreting $\beta_1$ as the causal effect:

$$\begin{align*} Y_{1,i} - Y_{0,i} &= \beta_0 + \beta_1 D_i +  \varepsilon _i  \\  
Y_{1,i}  &= \beta_0 + \beta_1 D_i + Y_{0,i} +  \varepsilon _i \end{align*} $$

<!-- ??? Warning inline end "Double Machine Learning Example"

    <figure markdown>
    ![Image title](./../fig/dml.png){ width="500" }
    <figcaption>Normalized Sampling Distribution</figcaption>
    </figure> -->
In practice, we don't usually condition on $X_i$ (which means there's no reason to correct for $\mathbb{P}(D|X)$, and this is very attractive because we know that the partially linear model fails to correct for this! --> See [Double Machine Learning Example](./examples.md))
and we allow 

$$Y_{0, i} \approx f(X_i, D_i) $$

But in some sense, this kinda feels like cheating because it suggests that de-panelled selection on observables is easier to work with than cross-sectional selection on observables. 



[^1]: Repeated cross-section