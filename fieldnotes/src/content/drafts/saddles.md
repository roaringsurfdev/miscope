
I'm seeing saddles at multiple scales.

The Global Parameter Space PCA for the MLP component appears to be tracing a curve on a saddle. The saddle-like shape was not apparent until PC3 was introduced.

When each frequency groups weight trajectory is projected onto the final basis for the Global Parameter Trajectory PCA, they each traverse a scaled down and minimally modulated shape of the global trajectory.

Frequency Group Centroids also appear to trace this same shape, but they're in different orientations. Procrustes analysis was used to show that each trajectory is similar. I'm not sure if this comparison was also done with the other two traces (Global Parameter Trajectory and Within Group Weight Trajectory).

The saddle found in within frequency group point cloud formations may be rotated with respect to the top 2 principal components. The reason I suspect this is that frequency 55 from p113/s999/ds598 is currently classified as a bowl, but both the point cloud manifold and the reconstructed surface appear to be saddles.

The 3rd lobe of the saddle in Parameter Trajectory PCA is the attractor. I think this saddle is rotated as well. The "bend" along the MLP global parameter trajectory corresponds with the dip (or rise, depending on arbitrary component signs) of one of the saddle axes.

What happens to the parameter trajectory visualization when we pin the basis at each epoch?
Relatedly, when does the parameter trajectory start looking like a saddle?
Extended training (from 25K to 35K) does not appear to dramatically change the shape of the trajectory, but it does seem to affec the PC2vPC3 plot. I'm wondering if the global saddle trajectory shows up around second descent, the way it does for the within-group manifolds.

Axes rise in within group manifolds once they reach a saturation/neuron count threshold.
Most groups achieve this level of saturation within second descent, but there are sometimes early slow risers or "first movers"

I find myself wanting to understand how the computation of a ball rolling on a rotating saddle might relate to network computations. And the obvious component involved in rotation is Attention. So I wonder if it owns some metaphorical/computational rotation/spin of the within group frequency clusters.

------------ additional notes ------------
I have noted that some within-group point cloud manifolds show "missing chunks". The missing chunks could correspond with the "remainder of the curve" that would trace the saddle if the trajectory hadn't exited the saddle.