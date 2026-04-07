## Roadmap - Analysis Rough Draft
*This is a landing place for a high-level collection of future **analysis** work. This is not meant to store future infrastructure ideas. It is not in any particular order and it is not meant to be authorative. As items are completed, they may be removed from this document.*

**Analysis Catalog**
With the addition of the view/catalog framework in the middle tier (miscope), it may be time to start formalizing and organizing the existing analysis. To support future use on new models (and to keep track of analysis on existing models), it would be useful to track the following fields for each view:
Name:
Category: [Summary, Parameter Space, Activation Space, Learned Representation] - can be multiple
Weight/Activation Spaces: [What portions of the network are being analyzed]
Purpose: [What does this analysis attempt to address]
Sources/Citations: [Where was this analysis used]
Math/Computation: [Any associated math, with formulas]
Description: [If someone else were to use the platform, this would allow them to gain a better understanding of this technique to determine whether they want to include it for a model or on a dashboard]
Implementation Details: [Completion status, Generic Analyzer->Renderer map, Possible Filters]

*Note: A markdown document (under articles/techniques) was created by Research Claude based on 10 techniques they are aware of and that are presented in a paper included in articles/releated_research*

**Animations**
At a high level, many of my requests are coming from the motivation to build animations for points of interest over training. I may also want to the ability to have a job queue for creating animations.

*Note: I can currently create animations using a purpose-built notebook.*

**Dominant Frequencies - All Weights**
We currently have an Analyzer called "Dominant Frequencies" that provides analysis on embedding frequencies. I would like to create a new analyzer that provides data on all weight spaces. At the very least, this should be expanded to include Output weights. We can leave the existing Analyzer as-is for now and create a new one if that's best. *Note: This is currently in progress*

**Cross-Variant Analysis**
Going to start needing some cross-variant comparison. This seems like it might be easiest for time/epoch series visualizations.

There are at least two ways of solving this, and both might be needed:
* Choose from list of variants for a model family to show on single epoch-series graph
* Side-by-side selection comparison where 2-4 models are selected, epoch-series data is combined where possible, then side-by-side visualizations where not. This would follow a product-comparison model selection UX.

**Mean IPR Analysis**
Add Mean IPR Analysis from notbooks/neuron_fourier_poc.py to analysis pipeline and renders.

**Per-frequency gradient energy per-checkpoint**
To be able to watch the accumulation trajectory from epoch 0 to cascade and see whether the winning frequencies had higher gradient energy from the start or whether they diverged later.

**Experiments for testing whether frequencies are a natural basis for neural networks**
p109/s485/ds598 shows geometric structure as early as epoch 500, which is during first descent and while test loss is still climbing. This really pushes against the theory that geometric formation is what the model is learning during generalization.
p101/s999/ds598 shows poor organization *and* poor MLP/Resid_Post Fourier Alignment throughout training.
Many models show early MLP/Resid_Post Fourier Alignment - as early as first descent.

In the October 2025 paper from Anthropic, (When Models Manipulate Manifolds: The Geometry of a Counting Task)[https://transformer-circuits.pub/2025/linebreaks/index.html], Gurnee et al., the model learns helical manifolds to compute line ends. These manifolds have periodicity *along them*. Here, in this modulo addition model, the internal geometry could have been a simple ring, but it's not. It's a ring with periodicity along the manifold. The explanation from the Anthropic paper is that the periodicity allows the model to differentiate near integer values along the manifold. I'm not sure I understand the value in this model. Why not just a 2D ring? Why all that extra structure?
What if the periodicity of the modulo addition model obscures the underlying need for models to leverage frequencies to carve up representation space? Or what if they serve some other organizing function? Human brains also use frequencies to coordinate neural populations, and they're capable of computing all kinds of non-periodic tasks. Why not neural networks, too, especially since they're inspired by human neural networks?

**Formalizing the Saddle Shape in Parameter Trajectory PCA**
All of the Modulo Addition Variants appear to traverse a saddle-like manifold in weight space over the course of training. 

**Formalizing the Saddle Shape in Neuron Frequency Group PCA**
There is another saddle shape showing up in the Neuron group PCA 3D Scatter (PC1xPC2xPC3). 
For p101/s485/ds598, all groups are intermingled.
For p101/s485/ds999, groups appear to be occupying independent saddle spaces, and Freq 28 appears heavily disorganized (hard to claim it as a saddle)
For p97/s42/ds598, two pairs of groups appear to be intermingled - a pair per saddle:(2, 22) and (28, 38). The two saddles appear to be orthogonal or near-orthogonal.
*Note: Saddle and Bowl geometries have been analytically confirmed quadratically using PCs. Next step is to surface these on the dashboard*

**Timaeus/SLT/Developmental Interpretability**
I've just become aware of the Timaeus project, and a technique called LLC estimation may be relevant to the saddles showing up in the network. Might be worth investigating to determine whether to incorporate analysis from their project into this one.

**Related Research on the Parameter Space Geometry**
It was brought to my attention that another independent researcher has been doing work that may overlap analysis from this platform. I've included two papers in notes/related_research: 2602.10496.pdf and 2602.16746.pdf

**Application of GLUE Framework to Weight Space**
*The first pass of this has been implemented*
Where Centroid Class PCA shows geometric organization, it seems that Frequency Group PCA (for MLPs) is showing us loss landscape traversal. It's also showing that transient frequencies may reside in separate basins from the the selected frequencies.

The existing Frequency Group PCA plots are useful in animations, but this leans on visual inspection and interpretation and is inaccessible to Claude for analysis.

I don't think it will be sufficient to create metrics against only one pair of PCs. Trajectories that are in separate spaces appear to cross in some PC pairs.

For Parameter Trajectory PCA there is a "Group Trajectory Proximity Plot (PC1/PC2, normalized)". This is a first step at a more rigorous visualization. 

For these PC plots (starting with Frequency Group PCA), I wonder if it's possible to create an elevation plot using PC3 upon which the PC1 and PC2 can be plotted.

Questions:
Does the first-mover frequency show early geometric organization? Answer: the PCA trajectories do not reveal this, but the quadratic curvature might.
Is grokking where the geometric organization happens in the *weights*? Answer: This is feeling more and more like a scale problem. What I mean by that is that organization in weight space is happening at different scales. At the macro scale, we see frequency groups traversing an apparent loss landscape. Within-group manifolds do appear to be showing geometric organization (manifold formation), though.

Is grokking the moment when the *weight geometry* finally catches up to the *activation geometry*? Answer: I still have no idea. Which is its own wonder that I just have to take a moment to humbly capture.

