## Roadmap Rough Draft
*This is a landing place for a high-level collection of future **infrastructure** work. This is not meant to store future research ideas. It is not in any particular order and it is not meant to be authorative. As items are completed, they may be removed from this document.*


### Context-Aware Navigation (context_aware_navigation.md)
It would be helpful to have a consistent left navigation that stays consistent across the application. The context of the left navigation can change based on context and it has expand/collapse behavior.

*Status: Work is complete and merged to develop*

### Sticky Analysis selections
When navigating through the "lenses"/pages of the application, I would like to keep the current Model Family, Variant, and Checkpoint constant across the pages.
*For now, this does not apply to the Analysis and Training sections. If these selections are sticky, that's ok, but this requirement does not need to include those pages* 

*Status: Work is complete and merged to develop*

### Consistent-Time Series Visualization Behavior
It is incredibly helpful to be able to click on a Visualization to navigate to an epoch of interest. For any time-series visualization with epochs along the x-axis, it would be ideal if they all shared the default behavior such that clicking on the visualization navigates to the epoch.

I'm not sure whether it would be necessary or useful to also provide the checkpoints on every visualization. I think this could either be informative or over-kill. Interpretation of time-series might change if the checkpoints are made clear.

*Note: There is an older requirement in the future bin that is related to this and may be outdated: REQ_014. I think it's actually completed and could be moved to the archive.*

*Status: Work is complete and merged to develop*

### Asynchronous Job Support
As we add more Variants and Analyzers, it would be nice to kick of Analysis runs as asynchronous jobs so that (1) the user can continue using the application and (2) the user can return to the Analysis Job Manager to check on the status of runs.

Once this capability is created for Analysis, it would be ideal to add Asynchronous Job Support for Training models. A use case for kicking off batches of Training runs would be if we wanted to add a new random seed across multiple Variants.

*Note: This behavior, along with general job management support will likely be built in a separate front-end from the analysis. The Dash framework may be ideal for rendering visualizations, but it might not be the right fit for job management. For ease of use, it may be that links are provided between front-ends.*

### Support for Choosing Subset for Analysis Runs
The Representational Geometry Analyzer is incredibly powerful and useful. However, it is a long-running Analyzer compared to the others. We may have other Analyzers that are similarly long-running. This might create friction for creating and exploring new Variants, since all Analyzers must run in order to Analyze a Variant. It would be nice to be able to choose which Analyzers to run.

### UI for Increasing/Descreasing Epoch Indexes
Right now, we have a slider and we have an ability to click on a graph to select a given epoch. Sometimes I'd like to select an epoch and the step forward or back one checkpoint at a time. Currently, the UI does not allow this, and I have to try to estimate or be extra careful with the slider. Adding Forward/Back or +/- buttons would help.

### Add support for Loading and Editing Variant Checkpoint Data
The current Training interface assumes I'm only creating new variants. I would like to be able edit an existing variant's checkpoints - for example, adding checkpoint for models whose grokking window falls outside the default grokking window. It would also be nice to be able to delete unecessary checkpoints and their artifacts through a single interface to make sure everything remains in sync. This requirement is coming from a need to rerun 107/485 with additional checkpoints. There were mistakes made on my end, and there's no way to gracefully recover without deleting and retraining/reanalyzing the entire model.

### Adding checkpoints should not need to retrain the whole model
If checkpoints are added to a model, the platform should be able to pick up with the last checkpoint before the newly added ones. If inserting checkpoints (adding density), it should only be necessary to train the model between known good checkpoints.