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

**Parameter Space PCA per epoch**
We currently show Parameter Space PCA for the entire training run. I'm wondering if it would make sense to create visualizations per epoch for side-by-side comparison with the Representation Space Centroid PCA diagrams. This would allow us to see whether there are dramatic moments during grokking, and whether the Parameter Space PCA undergoes a similar moment of using a 20/20/20 split between the first 3 PCs.

**Centroid Class PCA Distribution Dynamics**
The PCA redistributes across PC 1-3 over training. It would be nice to see this in a histogram or summary chart that shows the % of explanation of the top 3 PCs over training. This might be a useful indicator of a model undergoing reconstruction.

**Visibility into Neurons that Thrash**
It would be great to have a list of neurons that hop frequency specialization along with their respective concentrations of specialization. This would allow targeted analysis of specific neurons, especially in the context of lottery ticket analysis.

**Cross-Variant Analysis**
Going to start needing some cross-variant comparison. This seems like it might be easiest for time/epoch series visualizations.

There are at least two ways of solving this, and both might be needed:
* Choose from list of variants for a model family to show on single epoch-series graph
* Side-by-side selection comparison where 2-4 models are selected, epoch-series data is combined where possible, then side-by-side visualizations where not. This would follow a product-comparison model selection UX.

**Mean IPR Analysis**
Add Mean IPR Analysis from notbooks/neuron_fourier_poc.py to analysis pipeline and renders.
