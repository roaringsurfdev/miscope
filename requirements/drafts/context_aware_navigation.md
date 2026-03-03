## REQ_XXX Context-Aware Left Navigation
It would be helpful to have a consistent left navigation that stays consistent across the application. The context of the left navigation can change based on context and it has expand/collapse behavior.

### Training Job Menu
*Note: Job Management functionality will likely move to a separate application that's more suited to job management*
* **Select Model Family** (drop-down) - Selecting Model Family refreshes Variant List
* **Select Variant(s)** (Single-select list) - Selecting Variant(s) indicates which Variant Analysis jobs will be started
* **Create New/Edit/Save Button(s)** - CRUD buttons for Model Variant profiles. Note: Once a variant has been created, the only field that can be edited is the Checkpoint schedule. (NOTE: If the application doesn't yet support CRUD, it's possible to tackle this without a database backend since it's really just about running a new job with a different checkpoint schedule)
* **Start Job(s) Button** - This creates training job(s) based on selection

### Analyis Job menu
* **Select Model Family** (drop-down) - Selecting Model Family refreshes Variant List and Analyzer List
* **Select Variant(s)** (Multi-select list with option to Select All) - Selecting Variant(s) indicates which Variant Analysis jobs will be started
* **Select Analyzer(s)** (Multi-select list with option to Select All) - Selecting Analyzer(s) indicates which Analyzers to run in the new job that will be starts.
* **Start Job(s) Button** - This creates analysis job(s) based on selection

### Lenses 
(*A Lens is a page that has a collection of Visualizations that provide a lens into model mechanisms*)
* **Select Model Family** (drop-down) - Selecting Model Family refreshes Variant List
* **Select Variant(s)** (Single-select list) - Selecting Variant(s) indicates which Variant Analysis jobs will be started
* **Epoch Slider and Index Input** - Slider and input box combo that allows user to navigate epochs.
* **Lens-specific controls** - Section for lens-specific controls such as Attention Relationship, Trajectory Group, SV Matrix, Flatness Metric, (Neuron) Sort Order, Activation Site
