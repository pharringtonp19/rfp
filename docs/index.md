### **Perspective**

??? note "Concepts"
    
    - Attention v.s. Sampling 

From when I read Card and Kreuger 1994 in my second year of grad school, I've been interested in the relationship between "clusters" and selection on observables. The latter assumes a local level of variation, while the former prohibits it. 

At a high level, you can think of assuming selection on observables and then cluster-level assignment as transforming the probability measure of interest as follows.  

$$\mathbb{P} \mapsto G(\mathbb{P}) \mapsto H \circ G (\mathbb{P}) $$

Where we're interested in recovering $\theta(\mathbb{P})$ from $H \circ G (\mathbb{P})$


- Randomized Treatment
- Panel 
- Clusters
- Controls
- Policy 
- Propensity Score

``` mermaid
flowchart LR
    A(Randomized) -->|over|C;
    A(Randomized) -->|within|F;
    A(Randomized);
    B(Propesnity Score) --> |within|C;
    B(Propesnity Score) --> |within|F;
    C(Clusters);
    D(Pannel) --> A;
    F(Controls);
    F --> G(High);
    F --> Z(Low);

```

#### **Maybe?!**
Applied econometrics is all about interpretation. Specifically, it's about interpreting the result of a statistical procedure in a given context. 

The sobering reality of applied causal inference is that you often find yourself betting on something you don't believe is true, don't care about, or some combination of the two. No approach to causal inference can avoid this reality.

Taken together, the joke here is that applied econometrics is "easy" or at least requires little effort because this interpretation is always some flavor of "maybe". 

#### **Identification**

In observational studies, you are amost never identified underreasonable assumptions. 
```mermaid 
stateDiagram-v2
    state Justification {
    Identification --> Sample 
    Sample --> Estimator
    }
```

### **Principles**
!!! cite "The Bitter Lesson" 

    The lasting impression from Rich Sutton's [The Bitter Lesson](http://www.incompleteideas.net/IncIdeas/BitterLesson.html) is that one should step in the direction of greater compute. That is, one should work on methods that scale with greater computation.

    

!!! cite "Category Theory for Programmers" 

    Ideas should be explained at the level of detail so as to enable composition
   
