# REQ_001: Configurable Checkpoint Epochs

## Problem Statement
The training runner currently checkpoints at fixed intervals (every 100 epochs). For analyzing training dynamics, we need fine-grained control over checkpoint timing to capture more snapshots during critical phases (like grokking) and fewer during stable phases.

We need to be able to specify an integer list of epoch numbers where checkpoints should be saved, rather than using a fixed interval.

## Conditions of Satisfaction
- [x] Training accepts a list of epoch numbers for checkpointing (e.g., [100, 200, 500, 1000, 1500, 2000, ...])
- [x] Checkpoints are saved only at the specified epochs
- [x] If checkpoint list is not provided, falls back to reasonable default behavior
- [x] Checkpoint list can be configured per training run
- [x] Training loop efficiently handles arbitrary checkpoint spacing

## Constraints
**Must have:**
- Backward compatible with existing ModuloAdditionSpecification interface
- No performance degradation from checkpoint checking logic
- Works with immediate disk writes (REQ_002) - no memory accumulation

**Must avoid:**
- Creating checkpoints at unintended epochs
- Off-by-one errors in epoch numbering

**Flexible:**
- How checkpoint list is passed to training (constructor, method parameter, config object)
- Default checkpoint strategy if none specified

## Context & Assumptions
- Current implementation: ModuloAdditionSpecification.train() uses `checkpoint_every = 100`
- Current implementation accumulates checkpoints in memory (will be replaced by immediate disk writes in REQ_002)
- User wants to densify checkpoints around grokking phase based on prior knowledge of training dynamics
- REQ_002 handles the actual checkpoint persistence (separate files, safetensors)
- This requirement focuses on WHEN to checkpoint, REQ_002 handles HOW to persist
- Assumption: Epoch numbers in list are valid (< num_epochs)
- Assumption: Checkpoint list is sorted (or can be sorted automatically)

**Proposed default checkpoint strategy for ModuloAddition:**
```python
CHECKPOINTS = [
    *range(0, 1000, 100),      # Early training - sparse
    *range(1000, 5000, 500),   # Mid training - moderate
    *range(5000, 6000, 50),    # Grokking region - dense
    *range(6000, 10000, 500),  # Post-grokking - moderate
]
```
This strategy provides:
- ~10 checkpoints in early training (0-1000)
- ~8 checkpoints in mid training (1000-5000)
- ~20 checkpoints during grokking phase (5000-6000)
- ~8 checkpoints post-grokking (6000-10000)
- Total: ~46 checkpoints over 10000 epochs (vs 100 with fixed interval)

## Decision Authority
- [x] Make reasonable decisions and flag for review

## Success Validation
- Can train with custom checkpoint list: [100, 500, 1000, 2000, 5000, 10000]
- Verify correct number of checkpoint files created on disk
- Verify checkpoint epochs match provided list exactly
- Training with default behavior still works (reasonable default checkpoint schedule)
- No checkpoints created at unspecified epochs

---
## Notes

**Implementation considerations:**
- Default checkpoint list should be defined as a constant that can be easily modified
- Consider making the default strategy a function that generates checkpoints based on total epochs
- Grokking region (5000-6000) is specific to typical modular addition training; may need adjustment for other tasks
- Dense checkpointing in grokking region enables fine-grained analysis of circuit formation
- Reducing total checkpoints from 100 to ~46 saves disk space and analysis time while maintaining coverage

**Alternative approaches considered:**
- Logarithmic spacing: More principled but less intuitive, doesn't align with known grokking phases
- Fixed interval with manual additions: Less clean, harder to reason about coverage
- Adaptive checkpointing based on loss changes: More complex, requires online monitoring

---

## Implementation Notes (Added by Claude)

**Implementation completed:** 2026-01-31

**Approach taken:**
- Added `checkpoint_epochs` parameter to `train()` method (chosen over constructor to allow per-run configuration)
- Created `DEFAULT_CHECKPOINT_EPOCHS` constant at module level (46 checkpoints, matches proposed strategy)
- Checkpoint epochs are converted to a set for O(1) lookup during training loop
- Epochs >= num_epochs are automatically filtered out

**Key code locations:**
- `ModuloAdditionSpecification.py:17-22` - DEFAULT_CHECKPOINT_EPOCHS constant
- `ModuloAdditionSpecification.py:208-279` - Updated train() method with checkpoint_epochs parameter

**Tests:** `tests/test_checkpoint_and_persistence.py`
- `TestREQ001_ConfigurableCheckpointEpochs` class contains 7 tests covering all CoS items

**Design decisions:**
- Chose method parameter over constructor argument for flexibility (same spec can train with different schedules)
- Used set for checkpoint lookup to ensure no performance degradation even with large checkpoint lists
- Kept epoch numbering 0-indexed to match Python conventions and existing code
