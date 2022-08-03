## **Context**

To keep things simple, we describe our approach in the specific context of cluster-level randomized control trials where we're interested in estimating treatment heterogeneity.[^1] With a binary treatment variable such a problem can be formulated as below: 

$$\underset{f \in \sigma(x)}{\text{inf}} \ E\big[(Y - f)^2\big]$$
 
Such experiments are common in development, education, and health settings because they are:

- Generally easier to implement
- Better adhere to the potential outcome framework[^2]  
- Perhaps most importantly, allow us to understand the the effects of scaling the treatment.[^3]

## **Challenge (The Tragic Triad)[^4]**

Under the potential outcome framework, clustered level treatment assignment can be roughly thought of as forming the treatment and controls groups via random clustered sampling. From an estimation standpoint, this poses a few challenges because in each treatment group:

- We observe only a subset of the clusters
- The distribution of covariates can differ across clusters
- The distribution of outcomes conditional on covariates may differ across clusters

[^1]: Cluster-level randomized control trials are randomized control trials where treatment varies at a level above the unit of interest.

[^2]: Reduce the chance of spillover effects between treated and non-treated individuals

[^3]: Many large scale studies such as HIE prefer to include many control variables in their regression specification: size of family, age categories, education level, income, self-reported health status, and use of medical care in the year prior to the start of the experiment, kind of insurance (if any) the person had prior to the experiment, whether family members grew up in a city, suburb, or town, and spending on medical care and dental care prior to the experiment

[^4]: The expression "tragic triad" is taken from [Gradient Surgery for Multi-Task Learning](https://arxiv.org/pdf/2001.06782.pdf)