Key points about dashboard_temp:
* Uses new middle tier "views" to serve figures to dcc.Graph objects on the pages.
* Shows a proof-of-concept of behavior for empty vs populated visualizations.
* Refactors the layout structure for analysis pages: LeftNav.py now owns the left nav behavior (including expand/collapse, rendering the Variant Selector and Context-Aware nav elements) and Variant Selector owns logic on loading the selected Family, Variant, and Epoch and updating Variant State on the server.
* Analysis pages automatically load graphs with the selections from the Variant Selector.
* Support for designating some controls as "epoch-selectors" now exists. Clicking on an "epoch-selector" graph or moving the Variant Selector Epoch slider have the same result.
* Encapsulates sitenav logic into a single component

Still to be done:
* Create proof-of-concept wiring context-aware nav/filtering.

Implementation Decisions:
* Making sure the content loads when switching pages/lenses AND with the proper variant state values. A lot of work was done to avoid the "prevent_initial_call=True" flag on the pages (summary.py, neuron_dynamics.py, repr_geometry.py). Adding this flag to the page on_[page]_data_changed callbacks prevented the pages from loading content without making a forcing change to the variant parameters (such as the epoch).
* View logic has been pushed to the middle tier, which makes managing server state (VariantState) simpler.
* All pages currently have a _VIEW_LIST constant that defines the graph settings. This complicated overhead is necessary for supporting the click event capture on certain graphs. Graph Component IDs on Pages are eventually consumed by the Variant Selector to load a newly selected epoch and to update the epoch slider.
* Where possible, callbacks and layout and related functionality are stored within single files for the purposes of encapsulation. Where possible, it would be ideal to keep from maintaining all callbacks in a single file.
* Additional filtering using the context-aware left navigation sections have not yet been wired up.