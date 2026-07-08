# Bug Report: [BUG_NNN - Short Description]

## Discovery

**Date:** 2026-01-31

**Discovered by:** Human

**Context:** Loading the site for the first time

## Symptoms

An error displays in the top right corner of the main page:
'Value 512 is greater than maximum value 511.'


## Reproduction Steps

1. Launch the dashboard: `python -m dashboard.app`
2. Load the site: `http://localhost:7860`
3. 
4. 

**Reproduction rate:** [Always / Sometimes / Rare - include percentage if known]

## Expected Behavior

This doesn't seem to be affecting any observable behavior. There is another bug (BUG_002) that could be related, though.

## Actual Behavior
An error is generated in the web app output in the terminal:
```
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
  File "/home/.../projects/mechinterp/training-dynamics-workbench/.venv/lib/python3.13/site-packages/gradio/components/number.py", line 136, in raise_if_out_of_bounds
    raise Error(f"Value {num} is greater than maximum value {maximum}.")
gradio.exceptions.Error: 'Value 512 is greater than maximum value 511.'
```

## Environment

**Version/Commit:** First version of the front-end (REQ_010 addresses need for managing and displaying application version)

**Configuration:** [Any relevant config, flags, environment variables]

**Dependencies:** [Relevant library versions if applicable]

## Impact

**Severity:** Low

**User Impact:** Only 1 user

## Additional Context

[Any other relevant information: screenshots, logs, related issues, etc.]

## Notes

[Any initial observations or context that might be helpful]
