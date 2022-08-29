---
hide:
  - navigation
  - toc
---

<center>
# **Regularizing the Forward Pass**
</center>

<center>

<iframe src="https://slides.com/pharringtonp19/rtc/embed?token=FxFf2FUF&style=transparent" width="460" height="336" title="rtc" scrolling="no" frameborder="0" webkitallowfullscreen mozallowfullscreen allowfullscreen></iframe>

</center>

### **Perspective**
To some extent, we're all a bit like Richard McElreath (author of Rethinking Stastics) in that we've spent some time thinking about the statistics/ causal inference course we would have like to have in graduate school. 

??? tip "The Three Stages of Applied Econometric Maturity"

    1. As a fresh graudate student, your narrowly focused on identification. Your constantly thinking "Am I identified, under what assumptions am I identified?" 

    2. Later on in your time at grad school, you realize that this idea of "Identification" $+$ "Sample Analogy" is really just one way to answer the more general question an audience member/reader might ask -- "In order to believe your results, what am I betting on?" 

    3. Just before your graduate, though, you realize that this idea that your trying to convince someone to interpret your results in a similair way is absurd. What you "learn" from the data depends upon your initial beliefs (we're all Bayesians!) so to the aim cannot be to convince anyone. Rather, given that you don't have non-parametric identification, your facing a host of sampling issues, you want to walk your audience through the tradeoffs that you made in your stastical analysis so that they might learn something from your work (note this something might be very different from what you learned!)
 
### **Introduction**

Following the approach advocated in [Mostly Harmless Econometrics](https://www.mostlyharmlesseconometrics.com/), we view causal inference techniques as methods that aim to correct for the fact that the data is not generated from a randomized experiment.[^1] 
In this project, we specifically consider problems that can be thought of as "sampling" problems. That is, contexts where you have clusters of observations or where the proposenity-score is non-uniform. In this setting, the problem of correcting for the non-RCT natue of one's data becomes the challenge of implementing a "local" correction in a statistically and computationally reasonable way.



### **Approach**

We consider the following three sampling correction approaches. All of which can be understood as a gradient correction[^2]. As is the case with applied causal inference we don't select the best option so much as wel choose the "least" bad option![^3]

<center>

 Method | Challenge |
| --- | --- |
| Kernel Adjustments | Expensive $O(n^2)$ |
| Re-weighting | Generative Modeling | 
| Finetuning | Right Space | 

</center>





### **Preview of Results**

#### **Generalizing Across Clusters**
<center>
<img src="https://raw.githubusercontent.com/pharringtonp19/rfp/main/docs/fig/preview_results/grad_desc_toy_Standard%20(2).png" alt="drawing" width="350"/> 
<img src="https://raw.githubusercontent.com/pharringtonp19/rfp/main/docs/fig/preview_results/grad_desc_toy_MAML%20(1).png" alt="drawing" width="350"/> 
<img src="https://raw.githubusercontent.com/pharringtonp19/rfp/main/docs/fig/preview_results/grad_desc_toy_ESCluster%20(1).png" alt="drawing" width="350"/> 

</center>

[^1]: ??? tip "Probability Theory"

        We define our problem via a probability space with a probability measure that "corresponds" a randomized control trial.
        
        $$\big(\Omega, \mathcal{F}, \mathbb{P}_{\text{randomized}} \big)$$
        
        Given this probability space, we can define the parameter of interest as some function(al) of the probability measure. 

        $$\theta_0 = f(\mathbb{P}_{\text{randomized}} ) $$

        The non-randomized aspect of the data is then captured via some transformation $T$, such that 

        $$\mathbb{P}_{\text{observed}} = T(\mathbb{P}_{\text{randomized}} )$$

[^2]: ??? tip "Deep Learning"

        At a high level, a challenge in applied econometrics is to develop/propose/motivate methods that can make use of the ongoing work in deep learning. We remain model agonstic in the sense that our methods focus on gradient corrections (i.e. within training corrections) which is different from the post-training corrections as proposed in alternative methods like [Double Machine Learning](https://academic.oup.com/ectj/article/21/1/C1/5056401)
 
 [^3]: ??? tip "Applied Econometrics" 

        Applied econometrics is all about interpretation. Specifically, it's about interpreting the result of a statistical procedure in a given context. The sobering reality of applied causal inference is that you often find yourself betting on something you don't believe is true, don't care about, or some combination of the two. No approach to causal inference can avoid this reality. Taken together, the joke here is that applied econometrics is "easy" or at least requires little effort because this interpretation is always some flavor of "maybe". 