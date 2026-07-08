# Bug Report: [BUG_003 - Error when Index field is empty on Global Epoch Control panel]

## Discovery

**Date:** 2026-02-03

**Discovered by:** Human

**Context:** Manually entering an index into the Global Epoch Control box by deleting the existing value and entering a new one. EX: Delete '0' and enter '41'.

## Symptoms

[Describe what's going wrong. Be specific about observable behavior.]

- No UI symptom, only an error message in the server terminal window.
- All visualization boxes show 'Error'
- See error trace at the end of this document

## Reproduction Steps

1. Launch the dashboard: `python -m dashboard.app`
2. Load the site: `http://localhost:7860`
3. Navigate to analysis tab and select model that has been analyzed.
4. After visualizations load, attempt manually enter checkpoint index by deleting existing index and entering a new one.

**Reproduction rate:** Always, but the output might change.

## Expected Behavior

This seems to be caused by an event firing when the index text box is empty. This is avoidable and there are many workarounds, including refreshing the page and making sure there is always a value in the index text box.

## Actual Behavior

Clearing the index text box generates and error.

## Environment

**Version/Commit:** v0.1.2

**Configuration:** [Any relevant config, flags, environment variables]

**Dependencies:** [Relevant library versions if applicable]

## Impact

**Severity:** Low

**User Impact:** 1 User impacted minimally.

## Additional Context

[Any other relevant information: screenshots, logs, related issues, etc.]

## Notes

```
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
  File "/home/.../projects/mechinterp/training-dynamics-workbench/.venv/lib/python3.13/site-packages/gradio/components/slider.py", line 154, in preprocess
    Number.raise_if_out_of_bounds(payload, self.minimum, self.maximum)
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/.../projects/mechinterp/training-dynamics-workbench/.venv/lib/python3.13/site-packages/gradio/components/number.py", line 133, in raise_if_out_of_bounds
    if minimum is not None and num < minimum:
                               ^^^^^^^^^^^^^
TypeError: '<' not supported between instances of 'NoneType' and 'int'

```
