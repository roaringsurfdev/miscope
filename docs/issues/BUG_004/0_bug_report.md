# Bug Report: BUG_004 - Analysis Tab Variant Selection and Analysis Failures

## Discovery

**Date:** 2026-02-03

**Discovered by:** Human

**Context:** End-to-end testing on the new Analysis functionality

## Symptoms

It looks like there may be an issue with the connection between the list of variants and running analysis.
- Selecting a variant shows "No variant selected" in status bar
- Clicking "Run Analysis" returns quickly without doing anything
- Reloading the page causes a Gradio dropdown validation error

## Reproduction Steps

1. I trained the default variant of the Modulo Addition 1-layer model. Training seems to have completed successfully.
2. I switched to the Analysis tab and clicked "Refresh". The trained variant showed up in the drop-down list of variants.
3. When selecting the variant, nothing happened, and clicking "Run Analysis" did not appear to work.
4. Reloading the site caused an error message to appear in the top right corner. (See end of document for full error trace from the server)

**Reproduction rate:** Always (refreshing the site always produces the error in Step 4.)

## Expected Behavior

When selecting the Model Family trained Variant, I expect the status to update in the status bar and I expect to be able to Run Analysis if it hasn't run yet.

## Actual Behavior

Multiple issues were identified:
1. **Dropdown error on reload**: Gradio validates stored dropdown value against empty choices list
2. **Variant selection not working**: `state.selected_family_name` was `None` on page load
3. **Analysis using wrong paths**: `ModuloAdditionSpecification` expected `results/modulo_addition/` but family-based training saves to `results/modulo_addition_1layer/`

## Environment

**Version/Commit:** v0.1.3

**Configuration:** Used default training settings.

**Dependencies:** Gradio components/dropdown validation

## Impact

**Severity:** High

**User Impact:** 1 User

## Root Causes and Fixes

### Issue 1: Dropdown error on page reload
**Cause:** Variant dropdown initialized with `choices=[]` and populated via `app.load()`, but on reload Gradio validated stored value against empty choices before `init_variants` ran.
**Fix:** Initialize `initial_variant_choices` at app creation time and use it directly in dropdown creation.

### Issue 2: Variant selection showing "No variant selected"
**Cause:** `state.selected_family_name` was `None` on page load because `on_family_change` only fires on user interaction, not on initial load.
**Fix:** Pass `family_name` from dropdown as explicit parameter to `on_variant_change()` and `run_analysis_for_variant()`, use `effective_family_name = family_name or state.selected_family_name`.

### Issue 3: Analysis failing to find checkpoints
**Cause:** `run_analysis_for_variant()` used legacy `ModuloAdditionSpecification` which builds paths like `results/modulo_addition/modulo_addition_p113_seed999/` but new family-based training saves to `results/modulo_addition_1layer/modulo_addition_1layer_p113_seed999/`.
**Fix:** Created `VariantSpecificationAdapter` class that wraps a `Variant` and provides the interface expected by `AnalysisPipeline`, using the variant's correct paths.

## Files Modified
- `dashboard/app.py`: Fixed dropdown initialization, added family_name parameters
- `analysis/variant_adapter.py`: New adapter class bridging Variant to AnalysisPipeline
- `analysis/__init__.py`: Export VariantSpecificationAdapter

## Status: RESOLVED

## Notes
/home/.../projects/mechinterp/training-dynamics-workbench/.venv/lib/python3.13/site-packages/gradio/components/dropdown.py:235: UserWarning:

The value passed into gr.Dropdown() is not in the list of choices. Please update the list of choices to include: [('● prime=113, seed=999 [trained]', 'modulo_addition_1layer_p113_seed999')] or set allow_custom_value=True.

Traceback (most recent call last):
  File "/home/.../projects/mechinterp/training-dynamics-workbench/.venv/lib/python3.13/site-packages/gradio/queueing.py", line 766, in process_events
    response = await route_utils.call_process_api(
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    ...<5 lines>...
    )
    ^
  File "/home/.../projects/mechinterp/training-dynamics-workbench/.venv/lib/python3.13/site-packages/gradio/route_utils.py", line 355, in call_process_api
    output = await app.get_blocks().process_api(
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    ...<11 lines>...
    )
    ^
  File "/home/.../projects/mechinterp/training-dynamics-workbench/.venv/lib/python3.13/site-packages/gradio/blocks.py", line 2148, in process_api
    inputs = await self.preprocess_data(
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^
        block_fn, inputs, state, explicit_call
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    )
    ^
  File "/home/.../projects/mechinterp/training-dynamics-workbench/.venv/lib/python3.13/site-packages/gradio/blocks.py", line 1790, in preprocess_data
    processed_value = block.preprocess(inputs_cached)
  File "/home/.../projects/mechinterp/training-dynamics-workbench/.venv/lib/python3.13/site-packages/gradio/components/dropdown.py", line 207, in preprocess
    raise Error(
        f"Value: {value!r} (type: {type(value)}) is not in the list of choices: {choice_values}"
    )
gradio.exceptions.Error: "Value: ['● prime=113, seed=999 [trained]', 'modulo_addition_1layer_p113_seed999'] (type: <class 'list'>) is not in the list of choices: []"
