## Measurement Primitives
I think I'd like to consolidate REQs 097, 098, 104 into a single REQ. The central goal of this work is to properly define testable primitives that will allow for proper separation of concerns. If we view the work of the Analyzer pipeline through an ETL lens (see REQ_106), it's easier to see that there's really one REQ for pulling out and properly isolating existing measurement primitives:

For example:
	- Clustering: Centroids, radius, fisher discriminant, dimensionality
	- Shape fit: circle, sigmoid, Lissajous, saddle
	- Shape comparison/analysis: Procrustes
	- Fourier spectrum/decomposition
	- Velocity
	- PCA/Eigendecomposition (simple, summary/global trajectory, rolling)

These measurement primitives may already be encapsulated in some libraries, so it's important not to re-invent the wheel. If these functions exist in analysis libraries, what we likely need is tensor-friendly/ndarray-friendly versions of them. It also needs to be clear how each Analyzer hands data to them.


## Lakehouse opportunity
REQs 101 and 108 are both approaching a separate single requirement: output of tabular data from the Analyzers. There may also be some overlap with 107.

Right now, Analyzers primarily create blob storage along with some scalar/summary storage in json files. I'd like to explore leaning into a lakehouse pattern where some of the analyzers create parquet files in addition to or instead of npz files. This would open up more access to useable and query-able data. These files could be used in DataFrames as well as queried directly from an engine like DuckDB.

### PCA
Given the amoung of PCA applications across the platform, a first real win here would be to create tabular outputs of PCA such that each PCA operation generates two datasets:
- PCA Results
- PCA Projections

PCA Results Fields:
operation_type, variant_id, epoch, group, principal_component_index, eigenvalue/singular_value, eigenvector/basis_vector, explained_variance, explained_variance_ratio, participation_ratio, spread

PCA Projections Fields:
operation_type, variant_id, epoch, group, row_id, principal_component_index, principal_component_projection_value


Where "group" can be any of the following:
- single weight matrix
- weight component (combination of weight matrices for a component such as MLP, Attention)
- activation site
- single centroid
- centroid group for a given activation site

Where "operation_type" can be any of the following:
- Global Weight Trajectory (summary PCA) by Matrix or All Matrices
- Global Weight Velocity
- Ranged Global Weight Trajectory (summary PCA) by Matrix or All Matrices (only computes basis on subset(s) of epochs)
- Weight Matrix PCA (for Dimensionality, PR, Singular Values)
- Weight Matrix Velocity?
- Frequency Groups (Weight Matrices)
- Frequency Group Centroids
- Frequency Group PCA
- Frequency Group Trajectory (summary PCA)
- Frequency Group Velocity (?)
- Frequency Group Centroid Trajectory (summary PCA)
- Frequency Group Centroid Velocity (?)
- Frequency Group Shape Analysis
- Frequency Group Segment Analysis (bends in trajectories, shape of segments)
- Class Centroids (Activation Sites)
- Class Centroid PCA
- Class Centroid Trajectory (summary PCA)

### Fourier Analysis
A second real win would be to output Fourier Analysis into tabular data.

We currently measure frequencies across a weight group as well as frequencies for given neurons.

This section needs more concrete definition, but hopefully the idea comes through based on the PCA treatment.