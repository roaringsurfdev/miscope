# BUG_002 Investigation: Analysis Progress Bar Stuck at 10%

## Root Cause

The `run_analysis()` function in `dashboard/app.py` only had two progress update calls:
1. `progress(0, desc="Initializing analysis...")` - at initialization
2. `progress(0.1, desc="Running analysis pipeline...")` - before calling `pipeline.run()`

The `AnalysisPipeline.run()` method did not accept a progress callback, so the UI received no updates while the pipeline was processing checkpoints. The progress bar would stay at 10% until the entire analysis completed.

Additionally, Gradio's progress updates only render when the backend function yields or updates the progress object. Since `pipeline.run()` was a long-running synchronous operation without progress callbacks, the UI would freeze at 10%.

## Fix

Two changes were made:

### 1. `analysis/pipeline.py`
Added `progress_callback` parameter to `AnalysisPipeline.run()`:
```python
def run(
    self,
    epochs: list[int] | None = None,
    force: bool = False,
    save_every: int = 10,
    progress_callback: Callable[[float, str], None] | None = None,
) -> None:
```

The callback is invoked:
- Before processing each checkpoint (with progress 0.0 to 1.0)
- After all checkpoints are processed (with progress 1.0)

### 2. `dashboard/app.py`
Updated `run_analysis()` to pass a wrapper callback that maps pipeline progress (0-1) to UI progress (0.1-0.95):
```python
def pipeline_progress(pct: float, desc: str):
    # Map pipeline 0-1 to UI 0.1-0.95 (reserving 0-0.1 for init, 0.95-1.0 for finalize)
    ui_progress = 0.1 + (pct * 0.85)
    progress(ui_progress, desc=desc)

pipeline.run(progress_callback=pipeline_progress)
```

## Testing

- All 150 dashboard and pipeline tests pass
- Manual testing required to verify UI progress bar updates correctly

## Files Changed

- `analysis/pipeline.py` - Added `progress_callback` parameter and progress reporting
- `dashboard/app.py` - Updated `run_analysis()` to pass progress callback
