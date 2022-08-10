### **The Two Key Ideas**

1) We can implement any p.d kernel as the inner product in some high dimensional space 

$$k(x,z) = \langle \phi(x), \phi(z) \rangle _{\mathcal{H}}$$

2) Kernel Methods can "rationalize" linear models in higher dimensions

$$f(\theta, w, x) = w^T \phi_{\theta}(x) = \sum _i \alpha _i k(x, x_i)$$

!!! tip "RKHS"

    - Set: $\Omega$
    - Vector Space (of Functions): $F(\Omega, \mathbb{R})$
    - RKHS: $\mathcal{H} \subset \mathcal{F}$ 

        - $\mathcal{H}$ is a subspace of $F(\Omega, \mathbb{R})$
        - Endowed with a Hilbert Space structure
        - Evaluation Functional is continuous (bounded!)

        $$\begin{align*}E_x : \mathcal{H} \to \mathbb{R}, \quad 
        f \longmapsto f(x) \end{align*}$$

??? note Riesz 

    A hilbert space is isometric to its dual space

- Given a banach space, can we characterize the dual space (the space of continuous linear functionals). One example of this is that the dual space of the space of continuous linear functions on a compact space is the the space of borel measures on the comptact space 
- If we have a Hilbert Space, $\mathcal{H}$, we can characterize the dual space $\mathcal{H}^*$ as $\mathcal{H}$. And by this, we really mean that $\mathcal{H}$ is isometric to $\mathcal{H}^*$. 

$$\begin{align*} & \Lambda :: \mathcal{H} \to \mathcal{H} \to \mathbb{R} \\ 
& \Lambda \ y \ x = \langle x, y \rangle _{\mathcal{H}} \end{align*}$$

What's important to observe is that 
-  $\Lambda \ y \in \mathcal{H}^*$
- $\Lambda $ is a bijection

In a RKHS, the evaluation functionals as continuous, which means they are elements of the dual space, which means that they can be represented by elements in $\mathcal{H}$

$$E_x(f) = \langle f, k_x\rangle _{\mathcal{H}}$$

Note that the above provides a mapping from $\Omega$ to $\mathcal{H}$

$$\begin{align*}\phi &: \Omega \to \mathcal{H} \\
x &\longmapsto k_x, \quad \text{such that} \ E_x(f) = \langle f, k_x\rangle _{\mathcal{H}} \  \forall f \in \mathcal{H}\end{align*}$$

From this, we can construct a Kernel function, $K :: \ \Omega \to \Omega \to \mathbb{R}$, as follows:

$$\begin{align*}  
K \ x \ z &= \langle \phi(x), \phi(z) \rangle _{\mathcal{H}} \\
&=   \langle k_x, k_z \rangle _{\mathcal{H}} \\ 
&= E_z(k_x) \quad \text{by RKHS}\\ 
&= k_x(z) \end{align*}$$
 

 

