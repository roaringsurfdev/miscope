# v0.1.0 MVP Milestone Summary

**Released:** 2026-02-01
**Version:** 0.1.0
**Codename:** MVP (Minimum Viable Product)

## Overview

The Training Dynamics Workbench is a mechanistic interpretability research tool for analyzing how neural networks learn during training. This MVP release provides end-to-end capability to train Transformer models on modular addition, capture training snapshots, and visualize how computational structures emerge over time.

## Features Delivered

### Phase 1: Training Runner
| Requirement | Description |
|-------------|-------------|
| REQ_001 | Configurable checkpoint epochs with optimized default schedule |
| REQ_002 | Safetensors persistence replacing pickle format |

### Phase 2: Analysis Engine
| Requirement | Description |
|-------------|-------------|
| REQ_003 | Analysis pipeline architecture with resumable artifact generation |
| REQ_004 | Dominant embedding frequencies visualization |
| REQ_005 | Activation heatmaps visualization |
| REQ_006 | Neuron frequency cluster visualization |

### Phase 3: Workbench Dashboard
| Requirement | Description |
|-------------|-------------|
| REQ_007 | Gradio dashboard with training controls |
| REQ_008 | Analysis execution and synchronized visualization display |
| REQ_009 | Loss curves with epoch indicator |
| REQ_010 | Application versioning |

## Key Architectural Decisions

### Three-Layer Architecture
```
ModuloAdditionSpecification (Training)
         ↓ checkpoints/*.safetensors
AnalysisPipeline + Analyzers (Analysis)
         ↓ artifacts/*.npz
Visualization Renderers → Dashboard
```

### Artifact-Based Design
- **Rationale:** Separates expensive GPU computation from cheap visualization rendering
- **Benefit:** Fast iteration on visual parameters without recomputing analysis
- **Format:** NumPy compressed (.npz) for analysis artifacts

### Synchronized Epoch Slider
- **Key Innovation:** Single slider controls all four visualizations simultaneously
- **Purpose:** Enables correlation discovery ("When frequencies emerge at epoch X, what happens to neuron clusters?")

### Checkpoint Strategy
- 46 optimized checkpoints (vs. fixed-interval approach)
- Dense during grokking phase (9000-13000 epochs for p=113)
- Sparse before and after for efficiency

## File Locations

| Component | Location |
|-----------|----------|
| Training | `ModuloAdditionSpecification.py` |
| Analysis Pipeline | `analysis/pipeline.py` |
| Analyzers | `analysis/analyzers/` |
| Visualizations | `visualization/renderers/` |
| Dashboard | `dashboard/app.py` |
| Tests | `tests/` (156 tests, 8 files) |

## Bug Fixes Included

- **BUG_001:** Slider validation error on model load (off-by-one in neuron slider bounds)
- **BUG_002:** Analysis progress bar stuck at 10% (progress callback mapping issue)

## Test Coverage

- `test_checkpoint_and_persistence.py` - REQ_001, REQ_002
- `test_analysis_pipeline.py` - REQ_003 core infrastructure
- `test_dominant_frequencies_analyzer.py` - REQ_003 analyzer
- `test_remaining_analyzers.py` - REQ_003 analyzers
- `test_artifact_loader.py` - REQ_003 artifact loading
- `test_req_003_integration.py` - End-to-end integration
- `test_dashboard.py` - REQ_007, REQ_008, REQ_010
- `test_visualization_renderers.py` - REQ_004, REQ_005, REQ_006

## Archived Requirements

The following detailed requirement specifications are preserved in this directory:

- `REQ_001_configurable_checkpoint_epochs.md`
- `REQ_002_safetensors_persistence.md`
- `REQ_003_analysis_pipeline_architecture.md` (with sub-requirements)
- `REQ_004_dominant_embedding_frequencies_viz.md`
- `REQ_005_activation_heatmaps_viz.md`
- `REQ_006_neuron_frequency_clusters_viz.md`
- `REQ_007_gradio_dashboard_training_controls.md`
- `REQ_008_analysis_visualization_display.md`
- `REQ_009_loss_curves_epoch_indicator.md`
- `REQ_010_application_versioning.md`

These contain full implementation notes, conditions of satisfaction, and decision rationale from the development process.
