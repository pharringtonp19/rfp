## **Overview**

There is often a difference between the econometrics taught in graduate school courses and the econometrics emphasized in applied seminars. This difference is sometimes mistakenly attributed to a gap in mathematical backgrounds. The better explanation, though, is that the difference exist because in seminars, the data is messy: clusters of individuals receive the same treatment; people drop out of the sample; outcomes get censored; selection into treatment is unknown. 

With multiple issues jockeying for ``first-order`` consideration, it's rare that a pre-existing estimator addresses all of them. Because of this, it can be helpful to have econometric methods that are **well-targeted** (i.e. address a specific issue) and **composable** (i.e. the components fit together) so as to allow researchers to adjust their models for their specific context. With this aim in mind, we illustrate that a  regularized version of [MAML](https://arxiv.org/abs/1703.03400) offers a conceptually simply, model-agnostic way to adjust one's estimator for the presence of clustered data. Conceptually, the approach can be understood as a gradient correction that favors early stopping at the cluster level.


