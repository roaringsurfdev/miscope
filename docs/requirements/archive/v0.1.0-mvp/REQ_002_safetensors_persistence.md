# REQ_002: Safetensors Model Persistence

## Problem Statement
Current model persistence uses `torch.save()` which relies on pickle format. This has security concerns and version compatibility issues. The mechinterp research community has moved toward safetensors as the standard format.

Additionally, the current implementation accumulates all checkpoints in memory during training (appending to a list), which wastes memory and doesn't scale well.

We need to migrate checkpoint persistence to use safetensors for model weights, save each checkpoint as a separate file immediately upon creation, and maintain separate storage for training metadata (losses, indices, config).

## Conditions of Satisfaction
- [x] Each checkpoint saved as separate safetensors file immediately upon creation
- [x] Checkpoints written to disk during training (not accumulated in memory)
- [x] Directory structure organizes checkpoints and metadata clearly
- [x] Training metadata (train_losses, test_losses, train_indices, test_indices) saved separately (JSON or similar)
- [x] Model configuration saved in readable format
- [x] Can load individual checkpoint by epoch number
- [x] Backward compatible: can still load old pickle-based checkpoints for analysis

## Constraints
**Must have:**
- Safetensors for all model weights
- Separate file per checkpoint (each checkpoint is a complete, independently loadable model)
- Immediate disk writes (no accumulation in memory during training)
- TransformerLens compatibility (HookedTransformer can load from safetensors)
- Directory structure following pattern:
  ```
  results/
    {model_name}/
      {model_name}_p{prime}_seed{seed}/
        {model_name}_p(prime)_seed{seed}.safetensors (final model after training run)
        checkpoints/
          checkpoint_epoch_{epoch:05d}.safetensors
        artifacts/        # Analysis artifacts (created by REQ_003)
        metadata.json     # training losses, indices
        config.json       # model configuration
  ```

**Must avoid:**
- Pickle format for new checkpoints (security risk)
- Accumulating checkpoints in memory during training
- Breaking existing trained models (maintain load compatibility)

**Flexible:**
- Exact metadata format (JSON preferred, but YAML or similar acceptable)
- Exact directory naming convention (as long as it's clear and consistent)

## Context & Assumptions
- TransformerLens supports safetensors via HuggingFace transformers integration
- Current format: Single pickle file with dict containing model, config, checkpoints, losses, indices
- Current implementation accumulates checkpoints in memory: `model_checkpoints.append(copy.deepcopy(model.state_dict()))`
- Safetensors library is mature and well-supported
- Each checkpoint represents a complete model state suitable for independent analysis
- Assumption: Separate files enable better memory efficiency and independent checkpoint access

## Decision Authority
- [x] Make reasonable decisions and flag for review

## Success Validation
- Save training run with safetensors format
- Verify directory structure matches specification
- Monitor memory usage during training - should remain constant (not grow with checkpoints)
- Load checkpoint at specific epoch independently and verify model weights correct
- Load training metadata and verify losses/indices match original
- Attempt to load old pickle checkpoint and verify backward compatibility
- File inspection shows .safetensors extension (not .pth or .pkl) and proper directory organization

---
## Notes

## Implementation Notes (Added by Claude)

**Implementation completed:** 2026-01-31

**Directory structure implemented:**
```
results/
  modulo_addition/
    modulo_addition_p{prime}_seed{seed}/
      modulo_addition_p{prime}_seed{seed}.safetensors  # Final model
      checkpoints/
        checkpoint_epoch_00000.safetensors
        checkpoint_epoch_00100.safetensors
        ...
      artifacts/                    # Empty, for REQ_003
      metadata.json                 # train_losses, test_losses, indices, checkpoint_epochs
      config.json                   # Model architecture and training params
```

**Key code locations:**
- `ModuloAdditionSpecification.py:40-56` - Directory structure setup in `__init__`
- `ModuloAdditionSpecification.py:281-314` - Helper methods `_save_checkpoint`, `_save_config`, `_save_metadata`
- `ModuloAdditionSpecification.py:86-132` - Updated `load_from_file` with format detection
- `ModuloAdditionSpecification.py:134-188` - New methods `load_checkpoint`, `get_available_checkpoints`

**Tests:** `tests/test_checkpoint_and_persistence.py`
- `TestREQ002_SafetensorsPersistence` class contains 12 tests covering all CoS items
- Backward compatibility tested by creating legacy pickle files and verifying load works

**Design decisions:**
- `load_from_file()` auto-detects format: tries safetensors first, falls back to legacy pickle
- `load_checkpoint(epoch)` also supports legacy format for gradual migration
- Removed `copy` module import since checkpoints are no longer accumulated in memory
- Run name includes prime and seed for easy identification of different training runs

**Backward compatibility approach:**
- Legacy path (`results/modulo_addition/modulo_addition.pth`) preserved as fallback
- When both formats exist, new format takes precedence
- `load_checkpoint()` can retrieve from legacy format's in-memory checkpoint list
