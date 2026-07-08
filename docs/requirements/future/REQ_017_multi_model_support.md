# REQ_017: Multi-Model Support

**Status:** Future (needs workshopping)
**Priority:** High
**Estimated Effort:** High (dedicated mini-release)
**Related:** REQ_015 (Checkpoint Editor), REQ_016 (CUDA Compute)

## Problem Statement

The workbench currently supports only the Modulo Addition toy model. To be a general-purpose training dynamics workbench, it needs to support multiple toy models. Different models have different:
- Architectures (transformers, MLPs, CNNs, etc.)
- Training parameters (learning rate, batch size, epochs, etc.)
- Input/output shapes and semantics
- Interesting phenomena to observe (grokking, phase transitions, etc.)
- Relevant analysis types

The dashboard and analysis pipeline need to become model-agnostic while still supporting model-specific customizations.

## Conditions of Satisfaction

### Core Functionality
- [ ] Support for multiple toy model types
- [ ] Model-agnostic artifact storage and loading
- [ ] Model-agnostic analysis pipeline (where applicable)
- [ ] Model-specific visualizations where needed

### Model Registration
- [ ] Clear interface for adding new model types
- [ ] Model metadata schema (parameters, checkpoint structure, etc.)
- [ ] Model-specific training configuration
- [ ] Model-specific analysis configuration

### Dashboard Adaptation
- [ ] Model selector in dashboard
- [ ] Dynamic parameter controls based on selected model
- [ ] Appropriate visualizations for each model type
- [ ] Graceful handling of model-specific vs generic analyses

## Candidate Toy Models

Beyond Modulo Addition, potential models to support:

| Model | Phenomenon | Complexity |
|-------|------------|------------|
| Modulo Addition (current) | Grokking | Low |
| Parity | Phase transitions | Low |
| Sparse Parity | Feature learning | Medium |
| Induction Heads | In-context learning | Medium |
| Indirect Object Identification | Circuit formation | High |
| Othello | World models | High |

## Technical Considerations

### Current State (Modulo Addition)
- Model: `ModuloAdditionRefactored.py` - hardcoded architecture
- Training: Parameters specific to modulo addition
- Analysis: Some analyses are model-agnostic, others assume specific structure
- Dashboard: Hardcoded for modulo addition visualizations

### Abstraction Layers Needed

```
┌─────────────────────────────────────────────────────────┐
│                      Dashboard                          │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐     │
│  │  Generic    │  │   Model     │  │   Model     │     │
│  │  Controls   │  │  Selector   │  │  Specific   │     │
│  └─────────────┘  └─────────────┘  └─────────────┘     │
└─────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────┐
│                   Model Registry                        │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐     │
│  │  Modulo     │  │   Parity    │  │  Induction  │     │
│  │  Addition   │  │             │  │   Heads     │     │
│  └─────────────┘  └─────────────┘  └─────────────┘     │
└─────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────┐
│                  Analysis Pipeline                      │
│  ┌─────────────────────┐  ┌─────────────────────┐      │
│  │   Generic Analyses  │  │  Model-Specific     │      │
│  │   (loss, weights)   │  │  Analyses           │      │
│  └─────────────────────┘  └─────────────────────┘      │
└─────────────────────────────────────────────────────────┘
```

### Key Design Questions (Need Workshopping)

1. **Model Interface:**
   - What's the minimal interface a model must implement?
   - How do we handle models with different checkpoint contents?
   - How do we specify model-specific training parameters?

2. **Analysis Compatibility:**
   - Which analyses are truly model-agnostic?
   - How do we register model-specific analyses?
   - How do we handle analyses that need model-specific knowledge?

3. **Visualization Adaptation:**
   - Which visualizations work across models?
   - How do we specify model-specific visualizations?
   - How do axis labels, scales, etc. adapt per model?

4. **Parameter Schema:**
   - How do we define the parameter space for each model?
   - How does the checkpoint editor (REQ_015) adapt to different epoch ranges?
   - How do we validate model-specific configurations?

5. **Storage Organization:**
   - Directory structure: by model type, by experiment, or flat?
   - How do we prevent loading wrong model type's checkpoints?
   - Metadata schema for model identification?

## Constraints

**Must have:**
- Existing Modulo Addition functionality preserved
- Clear path for adding new models
- Model-agnostic core with model-specific extensions

**Must avoid:**
- Requiring code changes for every new model
- Explosion of model-specific code paths
- Breaking existing experiments/artifacts

**Flexible:**
- Number of models supported in initial release
- Depth of model-specific customization
- Whether to support custom user-defined models

## Context & Assumptions

- The workbench name is "Training Dynamics Workbench," not "Modulo Addition Workbench"
- Different toy models exhibit different interesting phenomena
- Some users will want to add their own models eventually
- Assumption: All supported models are small enough for interactive analysis

## Relationship to Other Requirements

- **REQ_015 (Checkpoint Editor):** Needs to handle different epoch ranges per model
- **REQ_016 (CUDA Compute):** Model training should be compute-backend agnostic
- **Future:** Custom model support could be a follow-on requirement

## Implementation Phases

### Phase 1: Model Abstraction
- Define model interface/protocol
- Refactor Modulo Addition to implement interface
- Abstract hardcoded model references in dashboard

### Phase 2: Registry System
- Implement model registry
- Model metadata schema
- Dynamic parameter controls in dashboard

### Phase 3: Second Model
- Add one additional toy model (suggest: Parity)
- Validate abstraction works
- Identify and fix model-specific assumptions

### Phase 4: Analysis Adaptation
- Categorize analyses as generic vs model-specific
- Implement analysis registration per model
- Adapt visualizations to model metadata

## Decision Authority

- [ ] User approval required before implementation
- [ ] Significant workshopping needed on model interface design
- [ ] Second model choice should be discussed

## Success Validation

- At least two models supported
- Adding a new model doesn't require dashboard changes
- Existing Modulo Addition workflows unchanged
- Clear documentation for adding new models

---

## Notes

**2026-02-01:** Requirement created as future work. Marked as needing workshopping—the model interface design is the critical decision that will shape the architecture.

**Key workshop questions:**
1. What does a model "plugin" look like?
2. How do we balance flexibility with simplicity?
3. Which analyses are truly universal vs model-specific?
4. Should we support user-contributed models, or curated set only?

**2026-02-21: Architectural reframe — read before workshopping**

The current draft frames multi-model support as: "which analyses cross over vs. which are model-specific," treating model-specific as a large first-class category to be registered and managed per family. Phase 4 reflects this ("Implement analysis registration per model").

This framing is now understood to be backwards. See [PROJECT.md](../../PROJECT.md) for the full architectural invariant. The revised position:

**Analytical views are universal instruments.** The category of truly model-specific analyses is much smaller than the current draft assumes. The real split is:

- **Universal analytical views** — applicable to any transformer: PCA (parameter space, representation space), Fourier analysis on weights, neuron activation patterns, attention patterns, loss curves, parameter velocity, effective dimensionality. These belong to the View Catalog and are available to all families automatically.
- **Task-specific performance views** — depends on what the model was trying to learn: accuracy against ground truth, per-class error rates, task-specific metrics. These are genuinely family-contributed.

What families contribute is *context*, not *views*: how to construct probes, interpretive metadata (e.g., a prime-based Fourier basis for modulo addition), and task-specific performance metrics.

The practical implication: when a new family is introduced, it gets the full analytical catalog for free. The family does not need to register views. Phase 4 of the current implementation plan should be rewritten accordingly — the question is not "which analyses to register per model" but "what context does this family provide that enriches the universal views."
