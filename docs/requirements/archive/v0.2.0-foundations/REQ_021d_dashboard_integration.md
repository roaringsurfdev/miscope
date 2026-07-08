# REQ_021d: Dashboard Integration

**Status:** Active
**Priority:** High
**Parent:** [REQ_021](REQ_021_model_families.md)
**Dependencies:** REQ_021a (Core Abstractions), REQ_021c (Modulo Addition Family)
**Last Updated:** 2026-02-03

## Problem Statement

The dashboard currently assumes a single model type. With the Model Family abstraction in place, the dashboard needs to:
1. Display available families for selection
2. Show variants within the selected family
3. Indicate variant state (untrained/trained/analyzed)
4. Preserve existing visualization functionality

## Scope

This sub-requirement covers:
1. Family selector component in dashboard
2. Variant listing with state indicators
3. Wiring visualizations to family-specific analyzers
4. Ensuring no regression in existing functionality

## Proposed Design

### Dashboard Flow

```
┌─────────────────────────────────────────────────────────────┐
│  Family: [Modulo Addition (1 Layer) ▼]                      │
├─────────────────────────────────────────────────────────────┤
│  Variants:                                                   │
│  ┌─────────────────────────────────────────────────────────┐│
│  │ ● p=113, seed=42    [Analyzed]   [View] [Re-analyze]   ││
│  │ ● p=97, seed=42     [Trained]    [Analyze]             ││
│  │ ○ p=113, seed=999   [Untrained]  [Train]               ││
│  └─────────────────────────────────────────────────────────┘│
├─────────────────────────────────────────────────────────────┤
│  Selected: p=113, seed=42                                    │
│  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐        │
│  │ Dominant     │ │ Neuron       │ │ Frequency    │        │
│  │ Frequencies  │ │ Activations  │ │ Clusters     │        │
│  └──────────────┘ └──────────────┘ └──────────────┘        │
│                                                              │
│  [Visualization Area]                                        │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### Component Structure

```python
# dashboard/components/family_selector.py
def create_family_selector(registry: FamilyRegistry) -> gr.Dropdown:
    """Create dropdown for family selection."""
    families = registry.list_families()
    choices = [(f.display_name, f.name) for f in families]
    return gr.Dropdown(
        choices=choices,
        label="Model Family",
        value=choices[0][1] if choices else None
    )

# dashboard/components/variant_list.py
def create_variant_list(
    family: ModelFamily,
    registry: FamilyRegistry
) -> gr.Dataframe:
    """Create variant listing with state indicators."""
    variants = registry.get_variants(family)
    data = [
        {
            "Parameters": format_params(v.params),
            "State": v.state.value,
            "Actions": get_available_actions(v.state)
        }
        for v in variants
    ]
    return gr.Dataframe(data)

# dashboard/components/visualization_tabs.py
def create_visualization_tabs(family: ModelFamily) -> gr.Tabs:
    """Create tabs for family's visualizations."""
    tabs = []
    for viz_name in family.visualizations:
        viz = VisualizationRegistry.get(viz_name)
        tabs.append(gr.Tab(label=viz.display_name))
    return gr.Tabs(tabs)
```

### State Indicators

| State | Icon | Available Actions |
|-------|------|-------------------|
| Untrained | ○ | Train |
| Trained | ● | Analyze, View Checkpoints |
| Analyzed | ● | View, Re-analyze, Compare |

### Event Flow

1. **Family Selection Changes:**
   - Update variant list for new family
   - Update visualization tabs for new family
   - Clear current visualization

2. **Variant Selection Changes:**
   - Load variant's available checkpoints
   - Update checkpoint slider range
   - Load/display first visualization

3. **Visualization Tab Changes:**
   - Load appropriate analyzer artifacts
   - Render visualization for current checkpoint

### Gradio Implementation Sketch

```python
def create_dashboard():
    registry = FamilyRegistry(Path("model_families"))

    with gr.Blocks() as app:
        # State
        current_family = gr.State(None)
        current_variant = gr.State(None)

        # Family Selection
        family_dropdown = create_family_selector(registry)

        # Variant List
        variant_table = gr.Dataframe(
            headers=["Parameters", "State", "Actions"],
            interactive=False
        )

        # Visualization Area
        with gr.Row():
            checkpoint_slider = gr.Slider(
                label="Checkpoint Epoch",
                minimum=0,
                maximum=100,
                step=1
            )

        viz_tabs = gr.Tabs()
        viz_output = gr.Plot()

        # Event Handlers
        family_dropdown.change(
            fn=on_family_change,
            inputs=[family_dropdown],
            outputs=[variant_table, viz_tabs, current_family]
        )

        variant_table.select(
            fn=on_variant_select,
            inputs=[variant_table, current_family],
            outputs=[checkpoint_slider, viz_output, current_variant]
        )

        checkpoint_slider.change(
            fn=on_checkpoint_change,
            inputs=[checkpoint_slider, current_variant, viz_tabs],
            outputs=[viz_output]
        )

    return app
```

## Conditions of Satisfaction

- [ ] Family dropdown populated from `FamilyRegistry`
- [ ] Selecting a family updates variant list
- [ ] Variant list shows state (untrained/trained/analyzed)
- [ ] Selecting a variant loads its checkpoints
- [ ] Visualization tabs reflect selected family's visualizations
- [ ] Checkpoint slider controls which epoch is visualized
- [ ] All existing visualizations work for `modulo_addition_1layer` family
- [ ] No regression: existing functionality preserved
- [ ] Graceful handling when no families/variants exist

## Constraints

**Must have:**
- Family selection persists during session
- Variant state reflects actual filesystem state
- Visualizations load without re-running analysis (use cached artifacts)

**Must avoid:**
- Breaking existing visualization rendering
- Hardcoding family-specific UI elements
- Blocking UI during artifact loading

**Flexible:**
- Exact layout and styling
- Whether to show "Create Variant" UI (can be post-MVP)
- Pagination strategy for many variants

## Testing Strategy

1. **Unit Tests:**
   - Family selector populates correctly
   - Variant list updates on family change
   - State indicators reflect filesystem state

2. **Integration Tests:**
   - Select family → select variant → view visualization
   - Switch families → verify clean state transition
   - Select analyzed variant → verify all visualizations render

3. **Manual Validation:**
   - Existing dashboard workflows still work
   - New family/variant UI is intuitive
   - Performance acceptable with multiple variants

## Notes

This sub-requirement intentionally defers:
- "Create New Variant" UI (variants created via CLI/scripts for now)
- Training progress display (existing status indicator acceptable)
- Cross-variant comparison views (future enhancement)

The goal is family-aware navigation without expanding dashboard scope.
