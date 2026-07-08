# REQ_007: Gradio Dashboard with Training Controls

## Problem Statement
The workbench needs a user interface to configure and launch training runs. Currently, training requires editing Python scripts and running them manually.

We need a Gradio dashboard that allows configuring model parameters, checkpoint settings, and initiating training runs through a web interface.

## Conditions of Satisfaction
- [x] Gradio interface with controls for key parameters:
  - Modulus (p) - integer input
  - Random seed - integer input
  - Data seed - integer input
  - Training fraction - slider (0.0 to 1.0)
  - Checkpoint epochs - text input for integer list
  - Save path - text input
- [x] Button to start training
- [x] Status indicator showing training progress (simple text or progress bar acceptable)
- [x] Feedback when training completes
- [x] Training runs without blocking the interface (or clear indication it's running)

## Constraints
**Must have:**
- Gradio framework for UI
- Ability to configure p (modulus) and seeds to match MVP success criteria
- Input validation for parameters (p > 0, valid checkpoint list format, etc.)

**Must avoid:**
- Complex real-time training visualization (deferred to post-MVP)
- Blocking UI during training (unless clearly communicated)

**Flexible:**
- UI layout and styling
- Advanced training parameters (learning rate, batch size, etc.) can be fixed/hidden for MVP
- Whether training runs synchronously with progress indicator or asynchronously in background
- Error handling sophistication (basic validation acceptable)

## Context & Assumptions
- ModuloAdditionSpecification handles training execution
- Training can take significant time (25k epochs)
- User has experience with trained model behavior (doesn't need to watch every epoch)
- Gradio supports both blocking and async operations
- Assumption: Running one training job at a time is acceptable for MVP

## Decision Authority
- [x] Make reasonable decisions and flag for review

## Success Validation
- Launch Gradio dashboard in browser
- Configure modulus p=113, seeds, and checkpoint list
- Start training run
- Training completes and saves checkpoints
- Can launch second training with different p value
- Saved artifacts are loadable by analysis pipeline

---
## Notes

## Implementation Notes (Added by Claude)

**Implementation completed:** 2026-01-31

**Key code location:**
- `dashboard/app.py` - Main Gradio application (Training tab)

**Training Tab UI:**
- Modulus (p): Number input with info tooltip
- Model Seed / Data Seed: Number inputs
- Training Fraction: Slider 0.1-0.9
- Total Epochs: Number input (default 25000)
- Checkpoint Epochs: Comma-separated text input
- Save Path: Text input (default "results/")
- Start Training button with progress indicator

**Implementation details:**
- Training runs synchronously with `gr.Progress()` status updates
- Input validation via `validate_training_params()` before execution
- Status text shows final losses and model save path on completion
- Integrates with `ModuloAdditionSpecification.train()`

**Tests:** 21 tests in `tests/test_dashboard.py`
