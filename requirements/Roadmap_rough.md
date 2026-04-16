## Roadmap Rough Draft
*This is a landing place for a high-level collection of future **infrastructure** work. This is not meant to store future research ideas. It is not in any particular order and it is not meant to be authorative. As items are completed, they may be removed from this document.*

### Asynchronous Job Support
As we add more Variants and Analyzers, it would be nice to kick of Analysis runs as asynchronous jobs so that (1) the user can continue using the application and (2) the user can return to the Analysis Job Manager to check on the status of runs.

Once this capability is created for Analysis, it would be ideal to add Asynchronous Job Support for Training models. A use case for kicking off batches of Training runs would be if we wanted to add a new random seed across multiple Variants.

*Note: This behavior, along with general job management support will likely be built in a separate front-end from the analysis. The Dash framework may be ideal for rendering visualizations, but it might not be the right fit for job management. For ease of use, it may be that links are provided between front-ends.*

### Analysis/Analyzer Config File
We started an Analysis Techniques catalog (`notes/analysis_techniques_catalog.md`) a while ago, and now that we're supporting two model types that might need different Analyzers, it will be helpful to convert this document to a first class part of the site. Along with the existing data that has been captured, it would be great if each Analyzer has an indicator for what model(s) it supports.

### Support for Choosing Subset for Analysis Runs
The current Analysis Run page on the dashboard runs *all* Analyzers on the selected Variant. I would like the option to run a subset of Analyzers on the selected Variant. This is useful in cases where we are adding a new Analyzer to the pipeline.

### Batch Analyzer to replace run_analysis.py notebook
Currently, the only way to run analysis across all Variants - or a subset of them - is to execute analysis from the run_analysis.py notebook. In an effort to clean up the codebase and reduce code surface area, I would like to replace the notebook with a page for running Analyzers in Batch. This page should allow the user to run a selection (or all) of the Analyzers against a selection (or all) of the Variants. Adding an option to configure a wait cycle between Analyzing each Variant is also desired to help prevent overheating when running on a single machine (current use case).


### Make the fieldnotes plots more readable for models and researchers
For plots on fieldnotes, pre-populate the export button with proper file naming to include information about the variant and the view. Consider a hidden image area that will be legible by models since Plotly visualizations are not.

### Add Prev/Next buttons to Neuron slider
Duplicate the prev/next buttons currently used on the Epoch slide for the Neuron ID.